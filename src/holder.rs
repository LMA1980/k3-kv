//! Core context memory holder state and shared memory arena management.
//!
//! Provides thread-safe arena allocation, mapping registry, deallocation,
//! and strict operator `LimitMEMLOCK` ceiling enforcement (28 GiB reference
//! board ceiling, ADR-037 / AGENTS.md Rule 5).

use core::mem::size_of;
use std::collections::HashMap;
use std::fmt;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use zerocopy::FromBytes;

use k3_core::K3Error;
use k3_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, BLOCK_SIZE_TOKENS, CACHELINE_ALIGNMENT, KvBlockMeta,
    OutputSlabHeader, SLAB_STATUS_IDLE, TAG_FREE, TokenInputSlotHeader,
};
use k3_core::ipc::protocol::{AllocateArenaRequest, StatusCode};
use k3_core::ipc::shm::{
    ArenaOptions, DEFAULT_LIMIT_MEMLOCK_BYTES, IpcError, SharedMemoryArena, system_page_size,
};

/// Sizing and geometry parameters for allocating a shared memory arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaAllocationConfig {
    /// Logical model identifier or name.
    pub model_id: String,
    /// Maximum context tokens (`n_ctx`).
    pub n_ctx: u32,
    /// Number of active slots (typically 1 for single-stream worker).
    pub active_slots: u32,
    /// Attention head dimension (e.g. 64, 128).
    pub head_dim: u32,
    /// Number of transformer layers in the model.
    pub num_layers: u32,
    /// Key-value head count (for GQA/MQA models, defaults to 2 if 0).
    pub head_count_kv: u32,
    /// Model vocabulary size (dimension of output logits, defaults to 32000 if 0).
    pub vocab_size: u32,
    /// Whether to mlock the mapped memory pages into physical RAM.
    pub lock_memory: bool,
    /// Explicit capacity override in bytes (e.g. from `AllocateArenaRequest`).
    pub explicit_bytes: Option<u64>,
    /// Explicit block count override (e.g. `max_blocks` from `AllocateArenaRequest`).
    pub max_blocks: Option<u32>,
}

impl ArenaAllocationConfig {
    /// Creates a new configuration with standard defaults for the given model dimensions.
    #[must_use]
    pub fn new(model_id: impl Into<String>, n_ctx: u32, head_dim: u32, num_layers: u32) -> Self {
        Self {
            model_id: model_id.into(),
            n_ctx,
            active_slots: 1,
            head_dim,
            num_layers,
            head_count_kv: 2,
            vocab_size: 32000,
            lock_memory: false,
            explicit_bytes: None,
            max_blocks: None,
        }
    }

    /// Derives configuration from an incoming [`AllocateArenaRequest`] message.
    #[must_use]
    pub fn from_request(req: &AllocateArenaRequest) -> Self {
        let n_ctx = if req.n_ctx > 0 {
            req.n_ctx
        } else if req.max_blocks > 0 {
            req.max_blocks * BLOCK_SIZE_TOKENS
        } else {
            64
        };
        Self {
            model_id: req.model_id.clone(),
            n_ctx,
            active_slots: 1,
            head_dim: 64,
            num_layers: 2,
            head_count_kv: 2,
            vocab_size: 32000,
            lock_memory: false,
            explicit_bytes: if req.arena_bytes > 0 {
                Some(req.arena_bytes)
            } else {
                None
            },
            max_blocks: if req.max_blocks > 0 {
                Some(req.max_blocks)
            } else {
                None
            },
        }
    }

    /// Sets the number of active slots.
    #[must_use]
    pub const fn with_active_slots(mut self, active_slots: u32) -> Self {
        self.active_slots = active_slots;
        self
    }

    /// Sets the key-value head count for GQA/MQA.
    #[must_use]
    pub const fn with_head_count_kv(mut self, head_count_kv: u32) -> Self {
        self.head_count_kv = head_count_kv;
        self
    }

    /// Sets the model vocabulary size for output logits.
    #[must_use]
    pub const fn with_vocab_size(mut self, vocab_size: u32) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Sets whether pages should be locked in RAM via `mlock`.
    #[must_use]
    pub const fn with_lock_memory(mut self, lock: bool) -> Self {
        self.lock_memory = lock;
        self
    }

    /// Sets an explicit capacity in bytes.
    #[must_use]
    pub const fn with_explicit_bytes(mut self, bytes: u64) -> Self {
        self.explicit_bytes = Some(bytes);
        self
    }

    /// Sets an explicit maximum block count.
    #[must_use]
    pub const fn with_max_blocks(mut self, max_blocks: u32) -> Self {
        self.max_blocks = Some(max_blocks);
        self
    }
}

/// Detailed layout offsets and capacity computed for an arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputedArenaLayout {
    /// Total capacity of the mapped shared memory arena in bytes (page aligned).
    pub total_bytes: u64,
    /// Total number of 16-token KV blocks allocated.
    pub block_count: u32,
    /// Byte offset to the [`KvBlockMeta`] table.
    pub meta_table_offset: u64,
    /// Byte offset to the token input slots region.
    pub token_slots_offset: u64,
    /// Byte offset to the KV cache block vector data region.
    pub kv_blocks_offset: u64,
    /// Byte offset to the output logit slabs region.
    pub output_slabs_offset: u64,
}

/// Aligns a 64-bit unsigned integer up to the nearest multiple of `align`.
const fn align_up_u64(val: u64, align: u64) -> u64 {
    let rem = val % align;
    if rem == 0 { val } else { val + (align - rem) }
}

/// Aligns a `usize` value up to the nearest multiple of `align`.
const fn align_up_usize(val: usize, align: usize) -> usize {
    let rem = val % align;
    if rem == 0 { val } else { val + (align - rem) }
}

/// Computes the sub-region layout offsets and page-aligned total byte capacity
/// for the specified arena configuration.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if any sizing parameter is zero or arithmetic overflows.
#[allow(clippy::too_many_lines)]
pub fn compute_arena_layout(
    config: &ArenaAllocationConfig,
) -> Result<ComputedArenaLayout, KvError> {
    if config.n_ctx == 0 {
        return Err(KvError::InvalidDimensions(
            "n_ctx must be greater than zero".to_owned(),
        ));
    }
    if config.head_dim == 0 {
        return Err(KvError::InvalidDimensions(
            "head_dim must be greater than zero".to_owned(),
        ));
    }
    if config.num_layers == 0 {
        return Err(KvError::InvalidDimensions(
            "num_layers must be greater than zero".to_owned(),
        ));
    }

    let active_slots = config.active_slots.max(1);
    let head_count_kv = config.head_count_kv.max(1);
    let vocab_size = config.vocab_size.max(1);

    // 1. Block count: 16 tokens per block
    let mut blocks_per_slot = config.n_ctx.div_ceil(BLOCK_SIZE_TOKENS);
    if let Some(mb) = config.max_blocks {
        blocks_per_slot = blocks_per_slot.max(mb);
    }
    let block_count = blocks_per_slot
        .checked_mul(active_slots)
        .ok_or_else(|| KvError::InvalidDimensions("block_count overflowed u32".to_owned()))?;

    // 2. Offsets:
    // ArenaHeader: 128 bytes (0..128)
    let meta_table_offset: u64 = 128;
    let meta_size = u64::from(block_count)
        .checked_mul(size_of::<KvBlockMeta>() as u64)
        .ok_or_else(|| KvError::InvalidDimensions("meta_table_size overflowed u64".to_owned()))?;
    let meta_end = meta_table_offset
        .checked_add(meta_size)
        .ok_or_else(|| KvError::InvalidDimensions("meta_end overflowed u64".to_owned()))?;

    // TokenInputSlot: cacheline aligned (64 bytes)
    let token_slots_offset = align_up_u64(meta_end, CACHELINE_ALIGNMENT as u64);
    let per_slot_bytes = u64::from(config.n_ctx)
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|t| t.checked_add(size_of::<TokenInputSlotHeader>() as u64))
        .ok_or_else(|| KvError::InvalidDimensions("token slot size overflowed u64".to_owned()))?;
    let aligned_slot_bytes = align_up_u64(per_slot_bytes, CACHELINE_ALIGNMENT as u64);
    let total_slots_bytes = aligned_slot_bytes
        .checked_mul(u64::from(active_slots))
        .ok_or_else(|| {
            KvError::InvalidDimensions("total token slots size overflowed u64".to_owned())
        })?;
    let token_slots_end = token_slots_offset
        .checked_add(total_slots_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("token_slots_end overflowed u64".to_owned()))?;

    // KvBlockData: cacheline aligned (64 bytes)
    let kv_blocks_offset = align_up_u64(token_slots_end, CACHELINE_ALIGNMENT as u64);
    // 2 vectors (Key & Value) per token * num_layers * head_count_kv * head_dim * sizeof(f32)
    let bytes_per_token = 2u64
        .checked_mul(u64::from(config.num_layers))
        .and_then(|v| v.checked_mul(u64::from(head_count_kv)))
        .and_then(|v| v.checked_mul(u64::from(config.head_dim)))
        .and_then(|v| v.checked_mul(size_of::<f32>() as u64))
        .ok_or_else(|| KvError::InvalidDimensions("bytes_per_token overflowed u64".to_owned()))?;
    let block_bytes = u64::from(BLOCK_SIZE_TOKENS)
        .checked_mul(bytes_per_token)
        .ok_or_else(|| KvError::InvalidDimensions("block_bytes overflowed u64".to_owned()))?;
    let total_kv_blocks_bytes =
        u64::from(block_count)
            .checked_mul(block_bytes)
            .ok_or_else(|| {
                KvError::InvalidDimensions("total_kv_blocks_bytes overflowed u64".to_owned())
            })?;
    let kv_blocks_end = kv_blocks_offset
        .checked_add(total_kv_blocks_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("kv_blocks_end overflowed u64".to_owned()))?;

    // OutputSlab: cacheline aligned (64 bytes)
    let output_slabs_offset = align_up_u64(kv_blocks_end, CACHELINE_ALIGNMENT as u64);
    let per_slab_bytes = u64::from(vocab_size)
        .checked_mul(size_of::<f32>() as u64)
        .and_then(|v| v.checked_add(size_of::<OutputSlabHeader>() as u64))
        .ok_or_else(|| KvError::InvalidDimensions("slab_bytes overflowed u64".to_owned()))?;
    let aligned_slab_bytes = align_up_u64(per_slab_bytes, CACHELINE_ALIGNMENT as u64);
    let total_slabs_bytes = aligned_slab_bytes
        .checked_mul(u64::from(active_slots))
        .ok_or_else(|| KvError::InvalidDimensions("total_slabs_bytes overflowed u64".to_owned()))?;
    let output_slabs_end = output_slabs_offset
        .checked_add(total_slabs_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("output_slabs_end overflowed u64".to_owned()))?;

    let min_required_bytes = output_slabs_end;
    let raw_bytes = if let Some(explicit) = config.explicit_bytes {
        explicit.max(min_required_bytes)
    } else {
        min_required_bytes
    };

    let page_size = system_page_size() as u64;
    let total_bytes = align_up_u64(raw_bytes, page_size);

    Ok(ComputedArenaLayout {
        total_bytes,
        block_count,
        meta_table_offset,
        token_slots_offset,
        kv_blocks_offset,
        output_slabs_offset,
    })
}

/// Metadata returned upon successful arena allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaAllocationResult {
    /// Unique identifier assigned to the allocated arena.
    pub arena_id: u64,
    /// Total capacity of the allocated arena in bytes (page aligned).
    pub total_bytes: u64,
    /// Total number of 16-token KV blocks initialized.
    pub block_count: u32,
    /// Model identifier associated with the arena.
    pub model_id: String,
}

/// Structured error type for KV holder operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// Memory allocation would exceed the operator `LimitMEMLOCK` ceiling.
    InsufficientLockableMemory {
        /// Requested byte capacity that could not be satisfied.
        required_bytes: u64,
        /// Current total allocated/locked bytes across active arenas.
        allocated_bytes: u64,
        /// Configured bounded memlock ceiling in bytes.
        memlock_ceiling_bytes: u64,
    },
    /// Requested arena ID was not found in the registry.
    ArenaNotFound(u64),
    /// Invalid geometry or configuration parameter.
    InvalidDimensions(String),
    /// Synchronization lock acquisition failure.
    LockError(String),
    /// Underlying IPC / OS operation failed (e.g. mmap, mlock).
    IpcError(String),
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientLockableMemory {
                required_bytes,
                allocated_bytes,
                memlock_ceiling_bytes,
            } => {
                write!(
                    f,
                    "[insufficient_lockable_memory] required {required_bytes} bytes exceeds remaining budget (allocated: {allocated_bytes}, ceiling: {memlock_ceiling_bytes})"
                )
            }
            Self::ArenaNotFound(id) => write!(f, "[arena_not_found] arena ID {id} not found"),
            Self::InvalidDimensions(msg) => write!(f, "[invalid_dimensions] {msg}"),
            Self::LockError(msg) => write!(f, "[lock_error] {msg}"),
            Self::IpcError(msg) => write!(f, "[ipc_error] {msg}"),
        }
    }
}

impl std::error::Error for KvError {}

impl KvError {
    /// Maps this error into its corresponding K3 protocol [`StatusCode`].
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        match self {
            Self::InsufficientLockableMemory { .. } => StatusCode::InsufficientLockableMemory,
            Self::ArenaNotFound(_) => StatusCode::ArenaNotFound,
            Self::InvalidDimensions(_) => StatusCode::CapabilityMismatch,
            Self::LockError(_) | Self::IpcError(_) => StatusCode::InternalError,
        }
    }

    /// Returns the associated process exit code.
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        self.status_code().as_exit_code()
    }
}

/// Synchronized internal registry and accounting state.
#[derive(Debug)]
struct HolderInner {
    /// Active shared memory arenas indexed by unique arena ID.
    arenas: HashMap<u64, Arc<SharedMemoryArena>>,
    /// Current total allocated bytes across all registered arenas.
    allocated_bytes: u64,
}

/// Core context memory holder state managing active shared memory arenas.
///
/// Designed per ADR-042 (policy-free memory holder) and ADR-037 / AGENTS.md Rule 5
/// (bounded `LimitMEMLOCK` ceiling enforcement).
#[derive(Debug)]
pub struct KvHolder {
    /// Bounded operator `LimitMEMLOCK` ceiling in bytes (default 28 GiB).
    memlock_ceiling_bytes: u64,
    /// Monotonically incrementing arena ID generator.
    next_arena_id: AtomicU64,
    /// Thread-safe registry and accounting state.
    inner: RwLock<HolderInner>,
}

impl Default for KvHolder {
    fn default() -> Self {
        Self::new()
    }
}

impl KvHolder {
    /// Creates a new `KvHolder` with the default operator `LimitMEMLOCK` ceiling (28 GiB).
    #[must_use]
    pub fn new() -> Self {
        Self::with_memlock_ceiling(DEFAULT_LIMIT_MEMLOCK_BYTES)
    }

    /// Creates a new `KvHolder` with an explicit bounded `LimitMEMLOCK` ceiling in bytes.
    #[must_use]
    pub fn with_memlock_ceiling(ceiling_bytes: u64) -> Self {
        Self {
            memlock_ceiling_bytes: ceiling_bytes,
            next_arena_id: AtomicU64::new(1),
            inner: RwLock::new(HolderInner {
                arenas: HashMap::new(),
                allocated_bytes: 0,
            }),
        }
    }

    /// Returns the configured `LimitMEMLOCK` ceiling in bytes.
    #[must_use]
    pub const fn memlock_ceiling_bytes(&self) -> u64 {
        self.memlock_ceiling_bytes
    }

    /// Returns total bytes currently allocated and tracked across all active arenas.
    #[must_use]
    pub fn total_allocated_bytes(&self) -> u64 {
        self.inner.read().map_or(0, |g| g.allocated_bytes)
    }

    /// Returns remaining lockable memory budget in bytes before reaching ceiling.
    #[must_use]
    pub fn available_lockable_bytes(&self) -> u64 {
        self.memlock_ceiling_bytes
            .saturating_sub(self.total_allocated_bytes())
    }

    /// Returns the count of currently registered active arenas.
    #[must_use]
    pub fn arena_count(&self) -> usize {
        self.inner.read().map_or(0, |g| g.arenas.len())
    }

    /// Returns `true` if an arena with the given ID exists.
    #[must_use]
    pub fn contains_arena(&self, arena_id: u64) -> bool {
        self.inner
            .read()
            .is_ok_and(|g| g.arenas.contains_key(&arena_id))
    }

    /// Returns a list of all currently active arena IDs.
    #[must_use]
    pub fn list_arena_ids(&self) -> Vec<u64> {
        self.inner
            .read()
            .map(|g| g.arenas.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Allocates and initializes a new per-model shared memory arena.
    ///
    /// Sizing is computed from `n_ctx`, `active_slots`, `head_dim`, and `num_layers`.
    /// Sizing is strictly accounted against the operator `LimitMEMLOCK` ceiling.
    ///
    /// # Errors
    ///
    /// - Returns [`KvError::InsufficientLockableMemory`] (exit code 11) if the requested
    ///   capacity would exceed the operator `LimitMEMLOCK` ceiling.
    /// - Returns [`KvError::InvalidDimensions`] if configuration geometry is invalid.
    /// - Returns [`KvError::IpcError`] if OS shared memory allocation or mapping fails.
    pub fn allocate(
        &self,
        config: &ArenaAllocationConfig,
    ) -> Result<ArenaAllocationResult, KvError> {
        let layout = compute_arena_layout(config)?;
        let arena_len_usize = usize::try_from(layout.total_bytes).map_err(|_| {
            KvError::InvalidDimensions("calculated arena size exceeds pointer width".to_owned())
        })?;

        let arena_id = self.next_arena_id.fetch_add(1, Ordering::Relaxed);
        let shm_name = format!("k3_kv_{}_{}", config.model_id, arena_id);

        // Pre-flight check and atomic reservation under write lock
        {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| KvError::LockError(e.to_string()))?;
            let current_allocated = guard.allocated_bytes;
            let projected = current_allocated.checked_add(layout.total_bytes).ok_or(
                KvError::InsufficientLockableMemory {
                    required_bytes: layout.total_bytes,
                    allocated_bytes: current_allocated,
                    memlock_ceiling_bytes: self.memlock_ceiling_bytes,
                },
            )?;

            if projected > self.memlock_ceiling_bytes {
                return Err(KvError::InsufficientLockableMemory {
                    required_bytes: layout.total_bytes,
                    allocated_bytes: current_allocated,
                    memlock_ceiling_bytes: self.memlock_ceiling_bytes,
                });
            }

            // Tentatively reserve bytes
            guard.allocated_bytes = projected;
        }

        // Allocate SharedMemoryArena
        let opts = ArenaOptions::new(&shm_name, arena_len_usize)
            .with_lock(config.lock_memory)
            .with_memlock_ceiling(self.memlock_ceiling_bytes);

        let mut arena = match SharedMemoryArena::create_with_options(opts) {
            Ok(arena) => arena,
            Err(err) => {
                // Rollback reservation on failure
                if let Ok(mut guard) = self.inner.write() {
                    guard.allocated_bytes =
                        guard.allocated_bytes.saturating_sub(layout.total_bytes);
                }
                return match err {
                    IpcError::InsufficientLockableMemory(k3e) => {
                        let req = match k3e {
                            K3Error::InsufficientLockableMemory { required_bytes, .. } => {
                                required_bytes
                            }
                            _ => layout.total_bytes,
                        };
                        Err(KvError::InsufficientLockableMemory {
                            required_bytes: req,
                            allocated_bytes: self.total_allocated_bytes(),
                            memlock_ceiling_bytes: self.memlock_ceiling_bytes,
                        })
                    }
                    other => Err(KvError::IpcError(other.to_string())),
                };
            }
        };

        // Initialize ArenaHeader, KvBlockMeta, and slot/slab headers
        if let Err(init_err) = initialize_arena_contents(&mut arena, config, &layout) {
            // Rollback reservation
            if let Ok(mut guard) = self.inner.write() {
                guard.allocated_bytes = guard.allocated_bytes.saturating_sub(layout.total_bytes);
            }
            return Err(init_err);
        }

        let actual_bytes = arena.len() as u64;
        let arc_arena = Arc::new(arena);

        // Commit arena to registry under write lock
        {
            let mut guard = self
                .inner
                .write()
                .map_err(|e| KvError::LockError(e.to_string()))?;
            // Adjust reservation if actual_bytes differs from layout.total_bytes (e.g. system page size rounding)
            if actual_bytes != layout.total_bytes {
                guard.allocated_bytes = guard
                    .allocated_bytes
                    .saturating_sub(layout.total_bytes)
                    .saturating_add(actual_bytes);
            }
            guard.arenas.insert(arena_id, arc_arena);
        }

        Ok(ArenaAllocationResult {
            arena_id,
            total_bytes: actual_bytes,
            block_count: layout.block_count,
            model_id: config.model_id.clone(),
        })
    }

    /// Convenience helper allocating an arena directly from high-level request parameters.
    ///
    /// # Errors
    /// Returns [`KvError`] on failure.
    pub fn allocate_arena(
        &self,
        model_id: &str,
        arena_bytes: u64,
        n_ctx: u32,
        max_blocks: u32,
    ) -> Result<ArenaAllocationResult, KvError> {
        let n_ctx = if n_ctx > 0 {
            n_ctx
        } else if max_blocks > 0 {
            max_blocks * BLOCK_SIZE_TOKENS
        } else {
            64
        };
        let mut config = ArenaAllocationConfig::new(model_id, n_ctx, 64, 2);
        if max_blocks > 0 {
            config = config.with_max_blocks(max_blocks);
        }
        if arena_bytes > 0 {
            config = config.with_explicit_bytes(arena_bytes);
        }
        self.allocate(&config)
    }

    /// Deallocates an arena by ID, reclaiming its allocated memory budget.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if the arena ID is not registered.
    pub fn deallocate(&self, arena_id: u64) -> Result<u64, KvError> {
        let mut guard = self
            .inner
            .write()
            .map_err(|e| KvError::LockError(e.to_string()))?;
        let arena = guard
            .arenas
            .remove(&arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        let freed_bytes = arena.len() as u64;
        guard.allocated_bytes = guard.allocated_bytes.saturating_sub(freed_bytes);
        Ok(freed_bytes)
    }

    /// Looks up an active arena by ID, returning a shared reference `Arc<SharedMemoryArena>`.
    #[must_use]
    pub fn get_arena(&self, arena_id: u64) -> Option<Arc<SharedMemoryArena>> {
        let guard = self.inner.read().ok()?;
        guard.arenas.get(&arena_id).cloned()
    }

    /// Obtains a duplicated file descriptor (`OwnedFd`) for the specified arena.
    ///
    /// Suitable for transfer via `SCM_RIGHTS` over Unix Domain Sockets.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if the arena is not found,
    /// or [`KvError::IpcError`] if `libc::dup` fails.
    pub fn get_arena_fd(&self, arena_id: u64) -> Result<OwnedFd, KvError> {
        let arena = self
            .get_arena(arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        arena
            .duplicate_fd()
            .map_err(|e| KvError::IpcError(e.to_string()))
    }

    /// Deallocates all registered arenas and resets allocated byte accounting to 0.
    pub fn clear(&self) -> usize {
        if let Ok(mut guard) = self.inner.write() {
            let count = guard.arenas.len();
            guard.arenas.clear();
            guard.allocated_bytes = 0;
            count
        } else {
            0
        }
    }
}

/// Initializes the `ArenaHeader`, `KvBlockMeta` entries, and slot/slab headers
/// within a newly mapped shared memory arena.
fn initialize_arena_contents(
    arena: &mut SharedMemoryArena,
    config: &ArenaAllocationConfig,
    layout: &ComputedArenaLayout,
) -> Result<(), KvError> {
    // 1. Initialize ArenaHeader at offset 0
    let total_bytes = arena.len() as u64;
    let hdr = arena
        .header_mut()
        .map_err(|e| KvError::IpcError(format!("failed to get header_mut: {e}")))?;
    *hdr = ArenaHeader {
        magic: ARENA_MAGIC,
        version: ARENA_VERSION,
        total_bytes,
        block_count: layout.block_count,
        tokens_per_block: BLOCK_SIZE_TOKENS,
        n_ctx: config.n_ctx,
        head_dim: config.head_dim,
        num_layers: config.num_layers,
        head_count_kv: config.head_count_kv.max(1),
        vocab_size: config.vocab_size.max(1),
        active_slots: config.active_slots.max(1),
        meta_table_offset: layout.meta_table_offset,
        token_slots_offset: layout.token_slots_offset,
        kv_blocks_offset: layout.kv_blocks_offset,
        output_slabs_offset: layout.output_slabs_offset,
        generation_counter: 1,
        _reserved: [0; 40],
    };
    hdr.validate()
        .map_err(|msg| KvError::InvalidDimensions(msg.to_string()))?;

    // 2. Initialize KvBlockMeta table (0..block_count)
    for idx in 0..layout.block_count as usize {
        let block_id = u32::try_from(idx)
            .map_err(|_| KvError::InvalidDimensions("block index exceeds u32 range".to_owned()))?;
        let meta = arena.block_meta_mut(idx).map_err(|e| {
            KvError::IpcError(format!("failed to initialize block_meta[{idx}]: {e}"))
        })?;
        *meta = KvBlockMeta {
            block_id,
            tag: TAG_FREE,
            _pad0: 0,
            refcount: 0,
            committed_token_count: 0,
            _pad1: 0,
            sampler_rng_state: [0; 4],
            padding: [0; 16],
        };
    }

    // 3. Initialize TokenInputSlotHeader for active_slots
    let active_slots = config.active_slots.max(1) as usize;
    let raw_slot_bytes =
        config.n_ctx as usize * size_of::<u32>() + size_of::<TokenInputSlotHeader>();
    let per_slot_bytes = align_up_usize(raw_slot_bytes, CACHELINE_ALIGNMENT);
    let token_slots_base = usize::try_from(layout.token_slots_offset).map_err(|_| {
        KvError::InvalidDimensions("token_slots_offset exceeds address space".to_owned())
    })?;
    for slot_idx in 0..active_slots {
        let slot_base = token_slots_base + slot_idx * per_slot_bytes;
        let slot_slice = arena
            .slice_mut(slot_base, size_of::<TokenInputSlotHeader>())
            .map_err(|e| {
                KvError::IpcError(format!(
                    "failed to slice token slot header[{slot_idx}]: {e}"
                ))
            })?;
        let token_hdr = TokenInputSlotHeader::mut_from_bytes(slot_slice).map_err(|e| {
            KvError::IpcError(format!(
                "failed to decode TokenInputSlotHeader[{slot_idx}]: {e}"
            ))
        })?;
        *token_hdr = TokenInputSlotHeader {
            slot_seq_id: 0,
            token_len: 0,
            start_position: 0,
        };
    }

    // 4. Initialize OutputSlabHeader for active_slots
    let vocab_size = config.vocab_size.max(1);
    let raw_slab_bytes = vocab_size as usize * size_of::<f32>() + size_of::<OutputSlabHeader>();
    let per_slab_bytes = align_up_usize(raw_slab_bytes, CACHELINE_ALIGNMENT);
    let output_slabs_base = usize::try_from(layout.output_slabs_offset).map_err(|_| {
        KvError::InvalidDimensions("output_slabs_offset exceeds address space".to_owned())
    })?;
    for slot_idx in 0..active_slots {
        let slab_base = output_slabs_base + slot_idx * per_slab_bytes;
        let slab_slice = arena
            .slice_mut(slab_base, size_of::<OutputSlabHeader>())
            .map_err(|e| {
                KvError::IpcError(format!(
                    "failed to slice output slab header[{slot_idx}]: {e}"
                ))
            })?;
        let slab_hdr = OutputSlabHeader::mut_from_bytes(slab_slice).map_err(|e| {
            KvError::IpcError(format!(
                "failed to decode OutputSlabHeader[{slot_idx}]: {e}"
            ))
        })?;
        *slab_hdr = OutputSlabHeader {
            request_id: 0,
            vocab_size,
            position: 0,
            status: SLAB_STATUS_IDLE,
            _reserved: [0; 44],
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation_and_header_initialization() {
        let holder = KvHolder::new();
        let config = ArenaAllocationConfig::new("test_model", 64, 64, 2)
            .with_active_slots(1)
            .with_head_count_kv(2)
            .with_vocab_size(32000);

        let res = holder.allocate(&config).expect("allocation should succeed");
        assert!(res.arena_id > 0);
        assert!(res.total_bytes >= 256 * 1024);
        assert_eq!(res.block_count, 4); // 64 tokens / 16 = 4 blocks
        assert_eq!(res.model_id, "test_model");

        // Accounting verification
        assert_eq!(holder.total_allocated_bytes(), res.total_bytes);
        assert_eq!(holder.arena_count(), 1);
        assert!(holder.contains_arena(res.arena_id));

        // Arena inspection
        let arena = holder.get_arena(res.arena_id).expect("arena lookup");
        let hdr = arena.header().expect("header access");
        assert_eq!(hdr.magic, ARENA_MAGIC);
        assert_eq!(hdr.version, ARENA_VERSION);
        assert_eq!(hdr.block_count, 4);
        assert_eq!(hdr.tokens_per_block, 16);
        assert_eq!(hdr.n_ctx, 64);
        assert_eq!(hdr.head_dim, 64);
        assert_eq!(hdr.num_layers, 2);
        assert_eq!(hdr.head_count_kv, 2);
        assert_eq!(hdr.vocab_size, 32000);
        assert_eq!(hdr.active_slots, 1);
        assert!(hdr.validate().is_ok());

        // Verify KvBlockMeta entries initialized to free
        for i in 0..4 {
            let meta = arena.block_meta(i).expect("block meta");
            assert_eq!(meta.block_id, u32::try_from(i).unwrap());
            assert_eq!(meta.tag, TAG_FREE);
            assert_eq!(meta.refcount, 0);
            assert_eq!(meta.committed_tokens(), 0);
        }
    }

    #[test]
    fn test_memlock_ceiling_rejection_single_arena() {
        // Configure small memlock ceiling (64 KiB)
        let small_ceiling = 64 * 1024;
        let holder = KvHolder::with_memlock_ceiling(small_ceiling);

        // Arena with n_ctx=64 requires ~256 KiB, exceeding 64 KiB ceiling
        let config = ArenaAllocationConfig::new("oversized_model", 64, 64, 2);
        let res = holder.allocate(&config);

        match res {
            Err(
                err @ KvError::InsufficientLockableMemory {
                    required_bytes,
                    allocated_bytes,
                    memlock_ceiling_bytes,
                },
            ) => {
                assert!(required_bytes > small_ceiling);
                assert_eq!(allocated_bytes, 0);
                assert_eq!(memlock_ceiling_bytes, small_ceiling);
                assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
                assert_eq!(err.exit_code(), 11);
            }
            other => panic!("Expected InsufficientLockableMemory, got {other:?}"),
        }

        // Verify no budget leaked
        assert_eq!(holder.total_allocated_bytes(), 0);
        assert_eq!(holder.arena_count(), 0);
    }

    #[test]
    fn test_memlock_ceiling_rejection_multi_arena_cumulative() {
        let config = ArenaAllocationConfig::new("model_a", 64, 64, 2);
        let layout = compute_arena_layout(&config).expect("layout");
        let single_arena_bytes = layout.total_bytes;

        // Ceiling sized to hold exactly 1 arena, but not 2
        let tight_ceiling = single_arena_bytes + (single_arena_bytes / 2);
        let holder = KvHolder::with_memlock_ceiling(tight_ceiling);

        // First allocation succeeds
        let res1 = holder.allocate(&config).expect("first allocation");
        assert_eq!(holder.total_allocated_bytes(), single_arena_bytes);

        // Second allocation must fail with exit code 11
        let config2 = ArenaAllocationConfig::new("model_b", 64, 64, 2);
        let res2 = holder.allocate(&config2);

        match res2 {
            Err(
                err @ KvError::InsufficientLockableMemory {
                    allocated_bytes,
                    memlock_ceiling_bytes,
                    ..
                },
            ) => {
                assert_eq!(allocated_bytes, single_arena_bytes);
                assert_eq!(memlock_ceiling_bytes, tight_ceiling);
                assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
                assert_eq!(err.exit_code(), 11);
            }
            other => panic!("Expected InsufficientLockableMemory, got {other:?}"),
        }

        // Verify allocated bytes remains strictly single_arena_bytes
        assert_eq!(holder.total_allocated_bytes(), single_arena_bytes);

        // Deallocate first arena
        let freed = holder.deallocate(res1.arena_id).expect("deallocate");
        assert_eq!(freed, single_arena_bytes);
        assert_eq!(holder.total_allocated_bytes(), 0);

        // Now second allocation succeeds cleanly
        let res3 = holder
            .allocate(&config2)
            .expect("second allocation succeeds after freed");
        assert_eq!(holder.total_allocated_bytes(), res3.total_bytes);
    }

    #[test]
    fn test_arena_lookup_and_fd_duplication() {
        use std::os::fd::AsRawFd;

        let holder = KvHolder::new();
        let config = ArenaAllocationConfig::new("model_fd", 64, 64, 2);
        let res = holder.allocate(&config).expect("allocation");

        // Lookup existing
        assert!(holder.get_arena(res.arena_id).is_some());
        let fd = holder.get_arena_fd(res.arena_id).expect("fd dup");
        assert!(fd.as_raw_fd() >= 0);

        // Lookup non-existent
        assert!(holder.get_arena(9999).is_none());
        assert_eq!(
            holder.get_arena_fd(9999).unwrap_err(),
            KvError::ArenaNotFound(9999)
        );
        assert_eq!(
            holder.get_arena_fd(9999).unwrap_err().status_code(),
            StatusCode::ArenaNotFound
        );
    }

    #[test]
    fn test_arena_deallocation_and_clear() {
        let holder = KvHolder::new();
        let config = ArenaAllocationConfig::new("model_clear", 64, 64, 2);
        let res1 = holder.allocate(&config).expect("alloc 1");
        let res2 = holder.allocate(&config).expect("alloc 2");
        assert_eq!(holder.arena_count(), 2);

        // Single deallocation
        let freed = holder.deallocate(res1.arena_id).expect("dealloc 1");
        assert_eq!(freed, res1.total_bytes);
        assert_eq!(holder.arena_count(), 1);
        assert!(!holder.contains_arena(res1.arena_id));
        assert!(holder.contains_arena(res2.arena_id));

        // Clear all
        let cleared = holder.clear();
        assert_eq!(cleared, 1);
        assert_eq!(holder.arena_count(), 0);
        assert_eq!(holder.total_allocated_bytes(), 0);
    }

    #[test]
    fn test_invalid_dimensions_rejected() {
        let holder = KvHolder::new();

        // n_ctx = 0
        let mut cfg = ArenaAllocationConfig::new("m", 0, 64, 2);
        assert_eq!(
            holder.allocate(&cfg).unwrap_err().status_code(),
            StatusCode::CapabilityMismatch
        );

        // head_dim = 0
        cfg = ArenaAllocationConfig::new("m", 64, 0, 2);
        assert_eq!(
            holder.allocate(&cfg).unwrap_err().status_code(),
            StatusCode::CapabilityMismatch
        );

        // num_layers = 0
        cfg = ArenaAllocationConfig::new("m", 64, 64, 0);
        assert_eq!(
            holder.allocate(&cfg).unwrap_err().status_code(),
            StatusCode::CapabilityMismatch
        );
    }
}
