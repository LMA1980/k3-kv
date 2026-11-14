//! Core context memory holder state and shared memory arena management.
//!
//! Provides thread-safe arena allocation, mapping registry, deallocation,
//! and strict operator `LimitMEMLOCK` ceiling enforcement (28 GiB reference
//! board ceiling, ADR-037 / AGENTS.md Rule 5).

use core::mem::{align_of, offset_of, size_of};
use std::collections::HashMap;
use std::fmt;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};
use std::sync::{Arc, RwLock};

use zerocopy::{FromBytes, IntoBytes};

use jero_core::K3Error;
use jero_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, BLOCK_SIZE_TOKENS, CACHELINE_ALIGNMENT, KvBlockMeta,
    OutputSlabHeader, RecurrentStateSlotHeader, SLAB_STATUS_IDLE, TAG_FREE, TokenInputSlotHeader,
    estimate_recurrent_state_bytes,
};
use jero_core::ipc::protocol::{AllocateArenaRequest, KV_DTYPE_F32, StatusCode};
use jero_core::ipc::shm::{
    ArenaOptions, DEFAULT_LIMIT_MEMLOCK_BYTES, IpcError, SharedMemoryArena, system_page_size,
};
use jero_core::{KvCacheView, KvDtype, KvGeometry, ScratchScope};

use crate::quant_kv::{VECTOR_PLANE_ALIGNMENT, pack_kv_batch_plane, unpack_kv_range_plane};

/// Reserved byte index inside [`ArenaHeader::_reserved`] indicating compressed `MLA` KV storage.
///
/// Positioned at index `8` after `kv_dtype` (`_reserved[0]`) and `seq_capacity` (`_reserved[4..8]`).
const ARENA_RESERVED_MLA_FLAG_OFFSET: usize = 8;

/// Maximum number of tokens tracked by each [`KvBlockMeta`] cacheline entry.
pub const KV_BLOCK_TOKEN_CAPACITY: usize = BLOCK_SIZE_TOKENS as usize;

/// Canonical error alias for [`ArenaKvWriter`], [`ArenaKvReader`], and KV holder operations.
pub type KvHolderError = KvError;

/// Per-layer KV cache retention policy for hybrid local/global attention models (ADR-027, ADR-037).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKvPolicy {
    /// Full-context global attention retaining up to `n_ctx` tokens.
    GlobalFullContext,
    /// Sliding-window local attention retaining at most `window_tokens` tokens.
    SlidingWindow {
        /// Maximum sliding window token span.
        window_tokens: u32,
    },
}

/// Computes total 16-token KV layer-blocks across all layers for a hybrid local/global schedule.
#[must_use]
pub fn compute_hybrid_kv_block_count(n_ctx: u32, layer_policies: &[LayerKvPolicy]) -> u32 {
    if n_ctx == 0 || layer_policies.is_empty() {
        return 0;
    }

    let mut total_blocks = 0_u32;
    for &policy in layer_policies {
        let layer_tokens = match policy {
            LayerKvPolicy::GlobalFullContext => n_ctx,
            LayerKvPolicy::SlidingWindow { window_tokens } => n_ctx.min(window_tokens),
        };
        let layer_blocks = if layer_tokens == 0 {
            0
        } else {
            layer_tokens.div_ceil(BLOCK_SIZE_TOKENS)
        };
        total_blocks = total_blocks.saturating_add(layer_blocks);
    }
    total_blocks
}

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
    /// Number of key-value attention heads per layer (GQA/MQA models).
    pub head_count_kv: u32,
    /// Model vocabulary size (dimension of output logits).
    pub vocab_size: u32,
    /// KV cache element encoding (`0 = F32`, `1 = F16`, `2 = Q8_0`).
    pub kv_dtype: u8,
    /// Whether to mlock the mapped memory pages into physical RAM.
    pub lock_memory: bool,
    /// Explicit capacity override in bytes (e.g. from `AllocateArenaRequest`).
    pub explicit_bytes: Option<u64>,
    /// Explicit block count override (e.g. `max_blocks` from `AllocateArenaRequest`).
    pub max_blocks: Option<u32>,
    /// Optional per-layer hybrid local/global KV retention schedule.
    pub layer_policies: Option<Vec<LayerKvPolicy>>,
    /// Optional compressed `MLA` per-token KV latent stride (`kv_lora_rank + qk_rope_head_dim`).
    pub kv_latent_dim: Option<u32>,
    /// Number of linear-time recurrent/SSM layers persisted via ping-pong banks.
    pub num_recurrent_layers: u32,
    /// Per-layer convolution history byte requirement in each recurrent bank.
    pub conv_state_bytes_per_layer: u32,
    /// Per-layer SSM/WKV recurrent state byte requirement in each recurrent bank.
    pub ssm_state_bytes_per_layer: u32,
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
            kv_dtype: KV_DTYPE_F32,
            lock_memory: false,
            explicit_bytes: None,
            max_blocks: None,
            layer_policies: None,
            kv_latent_dim: None,
            num_recurrent_layers: 0,
            conv_state_bytes_per_layer: 0,
            ssm_state_bytes_per_layer: 0,
        }
    }

    /// Creates a pure recurrent/SSM arena configuration (`O(1)` in `n_ctx`) with zero KV layers.
    #[must_use]
    pub fn new_recurrent(
        model_id: impl Into<String>,
        n_ctx: u32,
        vocab_size: u32,
        num_recurrent_layers: u32,
        conv_state_bytes_per_layer: u32,
        ssm_state_bytes_per_layer: u32,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            n_ctx,
            active_slots: 1,
            head_dim: 0,
            num_layers: 0,
            head_count_kv: 0,
            vocab_size,
            kv_dtype: KV_DTYPE_F32,
            lock_memory: false,
            explicit_bytes: None,
            max_blocks: None,
            layer_policies: None,
            kv_latent_dim: None,
            num_recurrent_layers,
            conv_state_bytes_per_layer,
            ssm_state_bytes_per_layer,
        }
    }

    /// Derives validated configuration from an incoming [`AllocateArenaRequest`] message.
    ///
    /// Fails closed with [`KvError::InvalidDimensions`] when any capacity or geometry
    /// dimension is zero, `kv_dtype` is unsupported, or dimensions exceed platform `usize`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `req.validate()` fails or any dimension
    /// overflows `usize`.
    pub fn from_request(req: &AllocateArenaRequest) -> Result<Self, KvError> {
        req.validate()
            .map_err(|msg| KvError::InvalidDimensions(msg.to_owned()))?;
        usize::try_from(req.n_ctx)
            .map_err(|_| KvError::InvalidDimensions("n_ctx overflowed usize".to_owned()))?;
        usize::try_from(req.max_blocks)
            .map_err(|_| KvError::InvalidDimensions("max_blocks overflowed usize".to_owned()))?;
        usize::try_from(req.num_layers)
            .map_err(|_| KvError::InvalidDimensions("num_layers overflowed usize".to_owned()))?;
        usize::try_from(req.head_count_kv)
            .map_err(|_| KvError::InvalidDimensions("head_count_kv overflowed usize".to_owned()))?;
        usize::try_from(req.head_dim)
            .map_err(|_| KvError::InvalidDimensions("head_dim overflowed usize".to_owned()))?;
        usize::try_from(req.vocab_size)
            .map_err(|_| KvError::InvalidDimensions("vocab_size overflowed usize".to_owned()))?;

        Ok(Self {
            model_id: req.model_id.clone(),
            n_ctx: req.n_ctx,
            active_slots: 1,
            head_dim: req.head_dim,
            num_layers: req.num_layers,
            head_count_kv: req.head_count_kv,
            vocab_size: req.vocab_size,
            kv_dtype: req.kv_dtype,
            lock_memory: false,
            explicit_bytes: Some(req.arena_bytes),
            max_blocks: Some(req.max_blocks),
            layer_policies: None,
            kv_latent_dim: None,
            num_recurrent_layers: 0,
            conv_state_bytes_per_layer: 0,
            ssm_state_bytes_per_layer: 0,
        })
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

    /// Sets per-layer hybrid local/global KV retention policies.
    #[must_use]
    pub fn with_layer_policies(mut self, policies: Vec<LayerKvPolicy>) -> Self {
        self.layer_policies = Some(policies);
        self
    }

    /// Configures compressed Multi-Head Latent Attention (`MLA`) per-token KV latent width (`kv_lora_rank + qk_rope_head_dim`).
    #[must_use]
    pub const fn with_kv_latent_dim(mut self, kv_lora_rank: u32, qk_rope_head_dim: u32) -> Self {
        self.kv_latent_dim = Some(kv_lora_rank.saturating_add(qk_rope_head_dim));
        self
    }

    /// Configures ping-pong recurrent/SSM state allocation on this arena (for pure SSM or hybrid models).
    #[must_use]
    pub const fn with_recurrent_state(
        mut self,
        num_recurrent_layers: u32,
        conv_state_bytes_per_layer: u32,
        ssm_state_bytes_per_layer: u32,
    ) -> Self {
        self.num_recurrent_layers = num_recurrent_layers;
        self.conv_state_bytes_per_layer = conv_state_bytes_per_layer;
        self.ssm_state_bytes_per_layer = ssm_state_bytes_per_layer;
        self
    }

    /// Sets the KV cache element encoding (`KV_DTYPE_F32`, `KV_DTYPE_F16`, or `KV_DTYPE_Q8_0`).
    #[must_use]
    pub const fn with_kv_dtype(mut self, kv_dtype: u8) -> Self {
        self.kv_dtype = kv_dtype;
        self
    }

    /// Estimates the `O(1)` double-buffered recurrent state byte requirement (`slot_0 + slot_1`),
    /// independent of `n_ctx`. Returns `0` when `num_recurrent_layers == 0`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `num_recurrent_layers > 0` with zero state bytes
    /// per layer, or if byte sizing overflows `u64`.
    pub fn estimate_recurrent_state_bytes(&self) -> Result<u64, KvError> {
        let (_, total) = compute_recurrent_bank_geometry(self)?;
        Ok(total)
    }
}

/// Detailed layout offsets and capacity computed for an arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputedArenaLayout {
    /// Total capacity of the mapped shared memory arena in bytes (page aligned).
    pub total_bytes: u64,
    /// Total number of 16-token KV blocks allocated.
    pub block_count: u32,
    /// Exact per-slot sequence capacity (`blocks_per_slot * 16`, or `0` when `num_layers == 0`).
    pub seq_capacity: u32,
    /// Byte stride of a single K or V position row (`KvDtype::row_bytes(kv_dim)`, `0` when `num_layers == 0`).
    pub kv_row_bytes: u64,
    /// 128-byte aligned byte stride of a single layer K or V plane (`0` when `num_layers == 0`).
    pub kv_plane_stride_bytes: u64,
    /// Byte offset to the [`KvBlockMeta`] table.
    pub meta_table_offset: u64,
    /// Byte offset to the token input slots region.
    pub token_slots_offset: u64,
    /// Byte offset to the KV cache block vector data region.
    pub kv_blocks_offset: u64,
    /// Byte offset to the output logit slabs region.
    pub output_slabs_offset: u64,
    /// Byte offset to the recurrent state region (after output slabs, or `0` when unused).
    pub recurrent_state_offset: u64,
    /// Byte stride of each ping-pong recurrent bank (`0` when unused).
    pub recurrent_bank_stride_bytes: u64,
    /// Total bytes reserved for the recurrent state header + two ping-pong banks (`0` when unused).
    pub recurrent_total_bytes: u64,
}

impl ComputedArenaLayout {
    /// Returns the 128-byte aligned byte offset of `layer`'s K plane within a homogeneous arena.
    #[must_use]
    pub const fn k_plane_offset(&self, layer: u32) -> u64 {
        self.kv_blocks_offset + (layer as u64) * 2 * self.kv_plane_stride_bytes
    }

    /// Returns the 128-byte aligned byte offset of `layer`'s V plane within a homogeneous arena.
    #[must_use]
    pub const fn v_plane_offset(&self, layer: u32) -> u64 {
        self.k_plane_offset(layer) + self.kv_plane_stride_bytes
    }

    /// Returns the 128-byte aligned byte offset of `layer`'s single latent plane when `MLA` is active.
    #[must_use]
    pub const fn mla_plane_offset(&self, layer: u32) -> u64 {
        self.kv_blocks_offset + (layer as u64) * self.kv_plane_stride_bytes
    }
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

const fn output_slab_stride_u64(vocab_size: u32) -> Option<u64> {
    let Some(raw) = (vocab_size as u64).checked_mul(size_of::<f32>() as u64) else {
        return None;
    };
    let Some(with_hdr) = raw.checked_add(size_of::<OutputSlabHeader>() as u64) else {
        return None;
    };
    Some(align_up_u64(with_hdr, VECTOR_PLANE_ALIGNMENT as u64))
}

fn compute_recurrent_bank_geometry(config: &ArenaAllocationConfig) -> Result<(u64, u64), KvError> {
    if config.num_recurrent_layers == 0 {
        return Ok((0, 0));
    }
    if config.conv_state_bytes_per_layer == 0 && config.ssm_state_bytes_per_layer == 0 {
        return Err(KvError::InvalidDimensions(
            "recurrent state bytes per layer must be greater than zero".to_owned(),
        ));
    }
    let align = CACHELINE_ALIGNMENT as u64;
    let conv_aligned = align_up_u64(u64::from(config.conv_state_bytes_per_layer), align);
    let ssm_aligned = align_up_u64(u64::from(config.ssm_state_bytes_per_layer), align);
    if let (Ok(conv_u32), Ok(ssm_u32)) = (u32::try_from(conv_aligned), u32::try_from(ssm_aligned)) {
        let total = estimate_recurrent_state_bytes(config.num_recurrent_layers, conv_u32, ssm_u32)
            .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
        let bank_stride = (total - (size_of::<RecurrentStateSlotHeader>() as u64)) / 2;
        return Ok((bank_stride, total));
    }
    let per_layer_stride = conv_aligned.checked_add(ssm_aligned).ok_or_else(|| {
        KvError::InvalidDimensions("recurrent per-layer stride overflowed u64".to_owned())
    })?;
    let bank_stride = u64::from(config.num_recurrent_layers)
        .checked_mul(per_layer_stride)
        .ok_or_else(|| {
            KvError::InvalidDimensions("recurrent bank_stride overflowed u64".to_owned())
        })?;
    let two_banks = 2_u64.checked_mul(bank_stride).ok_or_else(|| {
        KvError::InvalidDimensions("recurrent dual-bank size overflowed u64".to_owned())
    })?;
    let total_region = (size_of::<RecurrentStateSlotHeader>() as u64)
        .checked_add(two_banks)
        .ok_or_else(|| {
            KvError::InvalidDimensions("recurrent_total_bytes overflowed u64".to_owned())
        })?;
    Ok((bank_stride, total_region))
}

fn compute_per_layer_plane_row_bytes(kv_dim: u64, kv_dtype: KvDtype) -> Result<u64, KvError> {
    let kv_dim_usize = usize::try_from(kv_dim)
        .map_err(|_| KvError::InvalidDimensions("kv_dim overflowed usize".to_owned()))?;
    let row_usize = kv_dtype
        .row_bytes(kv_dim_usize)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    u64::try_from(row_usize)
        .map_err(|_| KvError::InvalidDimensions("kv_row_bytes overflowed u64".to_owned()))
}

/// Returns the token slot capacity bounded to [`BLOCK_SIZE_TOKENS`] for pure-recurrent (`O(1)`) arenas.
const fn effective_token_slot_capacity(config: &ArenaAllocationConfig) -> u32 {
    if config.num_layers == 0 && config.num_recurrent_layers > 0 {
        BLOCK_SIZE_TOKENS
    } else {
        config.n_ctx
    }
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
    if config.num_layers == 0 && config.num_recurrent_layers == 0 {
        return Err(KvError::InvalidDimensions(
            "num_layers or num_recurrent_layers must be greater than zero".to_owned(),
        ));
    }
    if config.num_layers > 0 && config.head_dim == 0 && config.kv_latent_dim.is_none() {
        return Err(KvError::InvalidDimensions(
            "head_dim must be greater than zero".to_owned(),
        ));
    }
    if config.kv_latent_dim == Some(0) {
        return Err(KvError::InvalidDimensions(
            "kv_latent_dim must be greater than zero".to_owned(),
        ));
    }
    let kv_dtype = KvDtype::from_u8(config.kv_dtype)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    if config.num_layers > 0 && config.kv_latent_dim.is_none() {
        if config.head_count_kv == 0 {
            return Err(KvError::InvalidDimensions(
                "head_count_kv must be greater than zero".to_owned(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(KvError::InvalidDimensions(
                "vocab_size must be greater than zero".to_owned(),
            ));
        }
    }

    let (recurrent_bank_stride_bytes, recurrent_total_bytes) =
        compute_recurrent_bank_geometry(config)?;

    let active_slots = config.active_slots.max(1);
    let head_count_kv = config.head_count_kv.max(1);
    let vocab_size = config.vocab_size.max(1);
    let plane_align = VECTOR_PLANE_ALIGNMENT as u64;

    // 1. Block count & per-slot seq_capacity: 16 tokens per block (0 for pure-recurrent O(1) arenas)
    let mut blocks_per_slot = if config.num_layers == 0 {
        0
    } else {
        config.n_ctx.div_ceil(BLOCK_SIZE_TOKENS)
    };
    if let Some(mb) = config.max_blocks {
        blocks_per_slot = blocks_per_slot.max(mb);
    }
    let block_count = blocks_per_slot
        .checked_mul(active_slots)
        .ok_or_else(|| KvError::InvalidDimensions("block_count overflowed u32".to_owned()))?;
    let seq_capacity = if config.num_layers == 0 {
        0
    } else {
        blocks_per_slot
            .checked_mul(BLOCK_SIZE_TOKENS)
            .ok_or_else(|| KvError::InvalidDimensions("seq_capacity overflowed u32".to_owned()))?
    };

    // 2. Offsets:
    // ArenaHeader: 128 bytes (0..128)
    let meta_table_offset: u64 = 128;
    let meta_size = u64::from(block_count)
        .checked_mul(size_of::<KvBlockMeta>() as u64)
        .ok_or_else(|| KvError::InvalidDimensions("meta_table_size overflowed u64".to_owned()))?;
    let meta_end = meta_table_offset
        .checked_add(meta_size)
        .ok_or_else(|| KvError::InvalidDimensions("meta_end overflowed u64".to_owned()))?;

    // TokenInputSlot: 128-byte vector aligned
    let token_slots_offset = align_up_u64(meta_end, plane_align);
    let slot_tokens = effective_token_slot_capacity(config);
    let per_slot_bytes = u64::from(slot_tokens)
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|t| t.checked_add(size_of::<TokenInputSlotHeader>() as u64))
        .ok_or_else(|| KvError::InvalidDimensions("token slot size overflowed u64".to_owned()))?;
    let aligned_slot_bytes = align_up_u64(per_slot_bytes, plane_align);
    let total_slots_bytes = aligned_slot_bytes
        .checked_mul(u64::from(active_slots))
        .ok_or_else(|| {
            KvError::InvalidDimensions("total token slots size overflowed u64".to_owned())
        })?;
    let token_slots_end = token_slots_offset
        .checked_add(total_slots_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("token_slots_end overflowed u64".to_owned()))?;

    // Contiguous layer-major KV slab `[slot][layer][K|V][pos][kv_dim]`: 128-byte vector aligned
    let kv_blocks_offset = align_up_u64(token_slots_end, plane_align);
    let (kv_row_bytes, kv_plane_stride_bytes, total_kv_blocks_bytes) = if config.num_layers == 0 {
        (0_u64, 0_u64, 0_u64)
    } else {
        let kv_row_bytes = if let Some(latent_dim) = config.kv_latent_dim {
            compute_per_layer_plane_row_bytes(u64::from(latent_dim), kv_dtype)?
        } else {
            let kv_dim = u64::from(head_count_kv)
                .checked_mul(u64::from(config.head_dim))
                .ok_or_else(|| KvError::InvalidDimensions("kv_dim overflowed u64".to_owned()))?;
            compute_per_layer_plane_row_bytes(kv_dim, kv_dtype)?
        };
        let planes_per_layer: u64 = if config.kv_latent_dim.is_some() { 1 } else { 2 };
        let raw_plane_bytes = u64::from(seq_capacity)
            .checked_mul(kv_row_bytes)
            .ok_or_else(|| {
                KvError::InvalidDimensions("kv_plane_bytes overflowed u64".to_owned())
            })?;
        let plane_stride = align_up_u64(raw_plane_bytes, plane_align);
        let total_kv_bytes = if let Some(policies) = config.layer_policies.as_deref() {
            let expected_layers = usize::try_from(config.num_layers).unwrap_or(0);
            if policies.len() != expected_layers {
                return Err(KvError::InvalidDimensions(format!(
                    "layer_policies length ({}) does not match num_layers ({expected_layers})",
                    policies.len()
                )));
            }
            let mut slot_kv_bytes = 0_u64;
            for &policy in policies {
                let layer_tokens = match policy {
                    LayerKvPolicy::GlobalFullContext => config.n_ctx,
                    LayerKvPolicy::SlidingWindow { window_tokens } => {
                        config.n_ctx.min(window_tokens)
                    }
                };
                let layer_seq_cap = if layer_tokens == 0 {
                    0_u64
                } else {
                    u64::from(layer_tokens.div_ceil(BLOCK_SIZE_TOKENS))
                        .checked_mul(u64::from(BLOCK_SIZE_TOKENS))
                        .ok_or_else(|| {
                            KvError::InvalidDimensions(
                                "hybrid layer_seq_cap overflowed u64".to_owned(),
                            )
                        })?
                };
                let layer_plane_raw = layer_seq_cap.checked_mul(kv_row_bytes).ok_or_else(|| {
                    KvError::InvalidDimensions("hybrid layer_plane_raw overflowed u64".to_owned())
                })?;
                let layer_plane_aligned = align_up_u64(layer_plane_raw, plane_align);
                let layer_bytes = planes_per_layer
                    .checked_mul(layer_plane_aligned)
                    .ok_or_else(|| {
                        KvError::InvalidDimensions("hybrid layer_bytes overflowed u64".to_owned())
                    })?;
                slot_kv_bytes = slot_kv_bytes.checked_add(layer_bytes).ok_or_else(|| {
                    KvError::InvalidDimensions("hybrid slot_kv_bytes overflowed u64".to_owned())
                })?;
            }
            u64::from(active_slots)
                .checked_mul(slot_kv_bytes)
                .ok_or_else(|| {
                    KvError::InvalidDimensions(
                        "hybrid total_kv_blocks_bytes overflowed u64".to_owned(),
                    )
                })?
        } else {
            let layer_bytes = planes_per_layer.checked_mul(plane_stride).ok_or_else(|| {
                KvError::InvalidDimensions("kv_layer_bytes overflowed u64".to_owned())
            })?;
            let slot_bytes = u64::from(config.num_layers)
                .checked_mul(layer_bytes)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("kv_slot_bytes overflowed u64".to_owned())
                })?;
            u64::from(active_slots)
                .checked_mul(slot_bytes)
                .ok_or_else(|| {
                    KvError::InvalidDimensions("total_kv_blocks_bytes overflowed u64".to_owned())
                })?
        };
        (kv_row_bytes, plane_stride, total_kv_bytes)
    };
    let kv_blocks_end = kv_blocks_offset
        .checked_add(total_kv_blocks_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("kv_blocks_end overflowed u64".to_owned()))?;

    // OutputSlab: 128-byte vector aligned
    let output_slabs_offset = align_up_u64(kv_blocks_end, plane_align);
    let aligned_slab_bytes = output_slab_stride_u64(vocab_size)
        .ok_or_else(|| KvError::InvalidDimensions("slab_bytes overflowed u64".to_owned()))?;
    let total_slabs_bytes = aligned_slab_bytes
        .checked_mul(u64::from(active_slots))
        .ok_or_else(|| KvError::InvalidDimensions("total_slabs_bytes overflowed u64".to_owned()))?;
    let output_slabs_end = output_slabs_offset
        .checked_add(total_slabs_bytes)
        .ok_or_else(|| KvError::InvalidDimensions("output_slabs_end overflowed u64".to_owned()))?;

    // RecurrentStateSlotHeader + dual ping-pong banks immediately after output slabs
    let recurrent_state_offset = if config.num_recurrent_layers > 0 {
        align_up_u64(output_slabs_end, plane_align)
    } else {
        0
    };

    let min_required_bytes = if config.num_recurrent_layers > 0 {
        recurrent_state_offset
            .checked_add(recurrent_total_bytes)
            .ok_or_else(|| {
                KvError::InvalidDimensions("arena min_required_bytes overflowed u64".to_owned())
            })?
    } else {
        output_slabs_end
    };
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
        seq_capacity,
        kv_row_bytes,
        kv_plane_stride_bytes,
        meta_table_offset,
        token_slots_offset,
        kv_blocks_offset,
        output_slabs_offset,
        recurrent_state_offset,
        recurrent_bank_stride_bytes,
        recurrent_total_bytes,
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
        let shm_name = format!("jero_kv_{}_{}", config.model_id, arena_id);

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
                    IpcError::InsufficientLockableMemory(err) => {
                        let req = match err {
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
    /// Allocates an arena directly from a validated [`AllocateArenaRequest`] payload.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] when geometry validation fails, or
    /// [`KvError::InsufficientLockableMemory`] when the memory ceiling is exceeded.
    pub fn allocate_from_request(
        &self,
        req: &AllocateArenaRequest,
    ) -> Result<ArenaAllocationResult, KvError> {
        let config = ArenaAllocationConfig::from_request(req)?;
        self.allocate(&config)
    }

    /// Allocates an arena from explicit model geometry and sizing parameters.
    ///
    /// Fails closed via [`AllocateArenaRequest::validate`] when any dimension or capacity is
    /// zero or when `kv_dtype` is unsupported.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if validation fails or [`KvError`] if allocation
    /// exceeds the configured `LimitMEMLOCK` ceiling.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate_arena(
        &self,
        model_id: &str,
        arena_bytes: u64,
        n_ctx: u32,
        max_blocks: u32,
        num_layers: u32,
        head_count_kv: u32,
        head_dim: u32,
        vocab_size: u32,
        kv_dtype: u8,
    ) -> Result<ArenaAllocationResult, KvError> {
        let req = AllocateArenaRequest {
            model_id: model_id.to_owned(),
            arena_bytes,
            n_ctx,
            max_blocks,
            num_layers,
            head_count_kv,
            head_dim,
            vocab_size,
            kv_dtype,
        };
        self.allocate_from_request(&req)
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

    /// Reads and validates the [`RecurrentStateSlotHeader`] for an active arena.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if `arena_id` is not registered, or
    /// [`KvError::InvalidDimensions`] / [`KvError::IpcError`] if the arena has no valid
    /// recurrent state region.
    pub fn read_recurrent_slot_header(
        &self,
        arena_id: u64,
    ) -> Result<RecurrentStateSlotHeader, KvError> {
        let arena = self
            .get_arena(arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        let arena_bytes = arena.as_slice();
        let hdr_bytes = arena_bytes
            .get(..size_of::<ArenaHeader>())
            .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
        let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
            .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
        let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
            KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
        })?;
        let region = arena_bytes.get(rec_off..).ok_or_else(|| {
            KvError::IpcError("arena truncated before recurrent region".to_owned())
        })?;
        read_recurrent_slot_header_from_region(region)
    }

    /// Reads the durably committed `active_bank` payload snapshot and header for crash recovery.
    ///
    /// # Errors
    /// Returns [`KvError::ArenaNotFound`] if `arena_id` is not registered, or
    /// [`KvError::InvalidDimensions`] / [`KvError::IpcError`] if the recurrent state region is invalid.
    pub fn read_active_recurrent_bank(
        &self,
        arena_id: u64,
    ) -> Result<(RecurrentStateSlotHeader, Vec<u8>), KvError> {
        let arena = self
            .get_arena(arena_id)
            .ok_or(KvError::ArenaNotFound(arena_id))?;
        let arena_bytes = arena.as_slice();
        let hdr_bytes = arena_bytes
            .get(..size_of::<ArenaHeader>())
            .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
        let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
            .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
        let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
            KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
        })?;
        let region = arena_bytes.get(rec_off..).ok_or_else(|| {
            KvError::IpcError("arena truncated before recurrent region".to_owned())
        })?;
        let rec_hdr = read_recurrent_slot_header_from_region(region)?;
        let total = usize::try_from(rec_hdr.total_region_bytes()).map_err(|_| {
            KvError::InvalidDimensions("recurrent region size exceeds address space".to_owned())
        })?;
        if region.len() < total {
            return Err(KvError::IpcError(
                "arena truncated inside recurrent ping-pong banks".to_owned(),
            ));
        }
        let bank_off =
            usize::try_from(rec_hdr.bank_offset_bytes(rec_hdr.active_bank)).map_err(|_| {
                KvError::InvalidDimensions("active bank offset exceeds address space".to_owned())
            })?;
        let bank_len = usize::try_from(rec_hdr.bank_stride_bytes).map_err(|_| {
            KvError::InvalidDimensions("bank stride exceeds address space".to_owned())
        })?;
        let bank_slice = &region[bank_off..bank_off + bank_len];
        Ok((rec_hdr, bank_slice.to_vec()))
    }
}

/// Computes the cacheline-aligned start offset of the ping-pong recurrent state region
/// deterministically immediately after the output logit slabs, without modifying [`ArenaHeader`].
#[must_use]
pub fn recurrent_region_offset(header: &ArenaHeader) -> u64 {
    let active_slots = u64::from(header.active_slots.max(1));
    let aligned_slab_bytes = output_slab_stride_u64(header.vocab_size.max(1)).unwrap_or(u64::MAX);
    let total_slabs_bytes = aligned_slab_bytes.saturating_mul(active_slots);
    let output_slabs_end = header.output_slabs_offset.saturating_add(total_slabs_bytes);
    align_up_u64(output_slabs_end, VECTOR_PLANE_ALIGNMENT as u64)
}

/// Reads and validates a [`RecurrentStateSlotHeader`] from the beginning of a recurrent region slice.
///
/// # Errors
/// Returns [`KvError::IpcError`] if `region` is shorter than 64 bytes or malformed, or
/// [`KvError::InvalidDimensions`] if header validation fails or `num_recurrent_layers == 0`.
pub fn read_recurrent_slot_header_from_region(
    region: &[u8],
) -> Result<RecurrentStateSlotHeader, KvError> {
    let hdr_bytes = region
        .get(..size_of::<RecurrentStateSlotHeader>())
        .ok_or_else(|| {
            KvError::IpcError("recurrent region shorter than 64-byte header".to_owned())
        })?;
    fence(Ordering::Acquire);
    let hdr = RecurrentStateSlotHeader::read_from_bytes(hdr_bytes)
        .map_err(|e| KvError::IpcError(format!("invalid RecurrentStateSlotHeader: {e}")))?;
    hdr.validate()
        .map_err(|msg| KvError::InvalidDimensions(msg.to_owned()))?;
    if hdr.num_recurrent_layers == 0 {
        return Err(KvError::InvalidDimensions(
            "recurrent state header reports zero recurrent layers".to_owned(),
        ));
    }
    Ok(hdr)
}

/// Returns `(header, active_bank_readonly, inactive_bank_mutable)` for zero-tearing ping-pong execution.
///
/// Workers read checkpointed previous state from the active bank slice and write next-step state
/// exclusively into the inactive staging bank slice prior to calling [`commit_recurrent_step_in_region`].
///
/// # Errors
/// Returns [`KvError::IpcError`] or [`KvError::InvalidDimensions`] if the region is truncated or invalid.
pub fn recurrent_banks_from_region_mut(
    region: &mut [u8],
) -> Result<(RecurrentStateSlotHeader, &[u8], &mut [u8]), KvError> {
    let hdr = read_recurrent_slot_header_from_region(region)?;
    let hdr_len = size_of::<RecurrentStateSlotHeader>();
    let total = usize::try_from(hdr.total_region_bytes()).map_err(|_| {
        KvError::InvalidDimensions("recurrent region size exceeds address space".to_owned())
    })?;
    let stride = usize::try_from(hdr.bank_stride_bytes).map_err(|_| {
        KvError::InvalidDimensions("recurrent bank_stride exceeds address space".to_owned())
    })?;
    if region.len() < total {
        return Err(KvError::IpcError(format!(
            "recurrent region len ({}) < required ({total})",
            region.len()
        )));
    }
    let dual_payload = &mut region[hdr_len..total];
    let (first_bank, second_bank) = dual_payload.split_at_mut(stride);
    if hdr.active_bank == 0 {
        Ok((hdr, &*first_bank, second_bank))
    } else {
        Ok((hdr, &*second_bank, first_bank))
    }
}

/// Atomically flips `active_bank` (`0 <-> 1`) and updates `committed_step` in a recurrent region header.
///
/// Emits a `Release` barrier after staging-bank writes and stores `(new_step, inactive_bank)` in a
/// single aligned 64-bit atomic store (`AtomicU64`) at byte offset `0` of `region`, guaranteeing that
/// either both fields advance together or neither changes on crash.
///
/// # Errors
/// Returns [`KvError::IpcError`] if `region` is unaligned (`% 8 != 0`) or truncated, or
/// [`KvError::InvalidDimensions`] if header invariants fail.
pub fn commit_recurrent_step_in_region(
    region: &mut [u8],
    new_step: u32,
) -> Result<RecurrentStateSlotHeader, KvError> {
    let mut hdr = read_recurrent_slot_header_from_region(region)?;
    if (region.as_ptr() as usize) % align_of::<AtomicU64>() != 0 {
        return Err(KvError::IpcError(
            "recurrent region base pointer is not 8-byte aligned for atomic commit".to_owned(),
        ));
    }
    let next_bank = hdr.inactive_bank();
    hdr.committed_step = new_step;
    hdr.active_bank = next_bank;

    let mut word = [0_u8; 8];
    word[0..4].copy_from_slice(&new_step.to_ne_bytes());
    word[4..8].copy_from_slice(&next_bank.to_ne_bytes());
    let packed = u64::from_ne_bytes(word);

    fence(Ordering::Release);
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "8-byte AtomicU64 alignment is runtime-checked directly above"
    )]
    // SAFETY: `region` has len >= 64 and its base pointer was verified 8-byte aligned (`AtomicU64`).
    unsafe {
        AtomicU64::from_ptr(region.as_mut_ptr().cast::<u64>()).store(packed, Ordering::Release);
    }

    Ok(hdr)
}

/// Locates the recurrent region inside `arena_bytes` via [`ArenaHeader`] and returns split ping-pong views.
///
/// # Errors
/// Returns [`KvError`] if [`ArenaHeader`] or the trailing recurrent region is invalid or truncated.
pub fn recurrent_banks_in_arena_mut(
    arena_bytes: &mut [u8],
) -> Result<(RecurrentStateSlotHeader, &[u8], &mut [u8]), KvError> {
    let hdr_bytes = arena_bytes
        .get(..size_of::<ArenaHeader>())
        .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
    let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
        .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
    let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
        KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
    })?;
    let region = arena_bytes
        .get_mut(rec_off..)
        .ok_or_else(|| KvError::IpcError("arena truncated before recurrent region".to_owned()))?;
    recurrent_banks_from_region_mut(region)
}

/// Locates the recurrent region inside `arena_bytes` via [`ArenaHeader`] and commits `new_step`.
///
/// # Errors
/// Returns [`KvError`] if [`ArenaHeader`] or the trailing recurrent region is invalid or truncated.
pub fn commit_recurrent_step_in_arena(
    arena_bytes: &mut [u8],
    new_step: u32,
) -> Result<RecurrentStateSlotHeader, KvError> {
    let hdr_bytes = arena_bytes
        .get(..size_of::<ArenaHeader>())
        .ok_or_else(|| KvError::IpcError("arena too small for ArenaHeader".to_owned()))?;
    let arena_hdr = ArenaHeader::read_from_bytes(hdr_bytes)
        .map_err(|e| KvError::IpcError(format!("invalid ArenaHeader: {e}")))?;
    let rec_off = usize::try_from(recurrent_region_offset(&arena_hdr)).map_err(|_| {
        KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
    })?;
    let region = arena_bytes
        .get_mut(rec_off..)
        .ok_or_else(|| KvError::IpcError("arena truncated before recurrent region".to_owned()))?;
    commit_recurrent_step_in_region(region, new_step)
}

fn populate_arena_header(
    hdr: &mut ArenaHeader,
    total_bytes: u64,
    config: &ArenaAllocationConfig,
    layout: &ComputedArenaLayout,
) -> Result<(), KvError> {
    let kv_dtype = KvDtype::from_u8(config.kv_dtype)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
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
        swa_window_size: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
        derived_weights_offset: 0,
        _reserved: [0_u8; 24],
    };
    hdr.set_kv_dtype(kv_dtype);
    hdr.set_seq_capacity(layout.seq_capacity);
    hdr._reserved[ARENA_RESERVED_MLA_FLAG_OFFSET] = u8::from(config.kv_latent_dim.is_some());
    hdr.validate()
        .map_err(|msg| KvError::InvalidDimensions(msg.to_string()))
}

/// Initializes the `ArenaHeader`, `KvBlockMeta` entries, and slot/slab headers
/// within a newly mapped shared memory arena.
#[expect(
    clippy::too_many_lines,
    reason = "initializes all arena structures in single sequential pass"
)]
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
    populate_arena_header(hdr, total_bytes, config, layout)?;

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
    let raw_slot_bytes = effective_token_slot_capacity(config) as usize * size_of::<u32>()
        + size_of::<TokenInputSlotHeader>();
    let per_slot_bytes = align_up_usize(raw_slot_bytes, VECTOR_PLANE_ALIGNMENT);
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
    let per_slab_bytes = output_slab_stride_u64(vocab_size)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| {
            KvError::InvalidDimensions("output slab byte size overflowed usize".to_owned())
        })?;
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

    // 5. Initialize RecurrentStateSlotHeader and zero dual ping-pong banks when configured
    if config.num_recurrent_layers > 0 {
        let align = CACHELINE_ALIGNMENT as u64;
        let conv_aligned = u32::try_from(align_up_u64(
            u64::from(config.conv_state_bytes_per_layer),
            align,
        ))
        .map_err(|_| {
            KvError::InvalidDimensions("conv_state_bytes_per_layer exceeds u32".to_owned())
        })?;
        let ssm_aligned = u32::try_from(align_up_u64(
            u64::from(config.ssm_state_bytes_per_layer),
            align,
        ))
        .map_err(|_| {
            KvError::InvalidDimensions("ssm_state_bytes_per_layer exceeds u32".to_owned())
        })?;
        let rec_off = usize::try_from(layout.recurrent_state_offset).map_err(|_| {
            KvError::InvalidDimensions("recurrent_state_offset exceeds address space".to_owned())
        })?;
        let rec_total = usize::try_from(layout.recurrent_total_bytes).map_err(|_| {
            KvError::InvalidDimensions("recurrent_total_bytes exceeds address space".to_owned())
        })?;
        let rec_slice = arena.slice_mut(rec_off, rec_total).map_err(|e| {
            KvError::IpcError(format!("failed to slice recurrent state region: {e}"))
        })?;
        rec_slice.fill(0);
        let rec_hdr = RecurrentStateSlotHeader::new(
            config.num_recurrent_layers,
            conv_aligned,
            ssm_aligned,
            layout.recurrent_bank_stride_bytes,
        );
        rec_hdr
            .validate()
            .map_err(|msg| KvError::InvalidDimensions(msg.to_owned()))?;
        let hdr_size = size_of::<RecurrentStateSlotHeader>();
        rec_slice[..hdr_size].copy_from_slice(rec_hdr.as_bytes());
    }

    Ok(())
}

/// Returns the exact per-slot sequence capacity (`blocks_per_slot * 16`) persisted in
/// [`ArenaHeader`], avoiding slab-padding drift.
#[must_use]
pub fn arena_seq_capacity(header: &ArenaHeader) -> u32 {
    let stored = header.seq_capacity();
    if stored > 0 {
        return stored;
    }
    let slots = header.active_slots.max(1);
    (header.block_count / slots).saturating_mul(header.tokens_per_block)
}

/// Returns the KV cache element encoding (`KV_DTYPE_*`) recorded in [`ArenaHeader`].
#[must_use]
pub const fn arena_kv_dtype(header: &ArenaHeader) -> u8 {
    header._reserved[0]
}

/// Typed shared-memory handle exposing validated, bounds-checked views into an initialized arena.
#[derive(Debug)]
pub struct ArenaHandle<'a> {
    arena: &'a mut SharedMemoryArena,
}

impl<'a> ArenaHandle<'a> {
    /// Wraps a mapped [`SharedMemoryArena`] after validating its [`ArenaHeader`].
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] when the mapped buffer is smaller than an
    /// [`ArenaHeader`] or fails [`ArenaHeader::validate`].
    pub fn new(arena: &'a mut SharedMemoryArena) -> Result<Self, KvError> {
        {
            let (header, _) = ArenaHeader::ref_from_prefix(arena.as_slice()).map_err(|_| {
                KvError::InvalidDimensions("arena buffer smaller than ArenaHeader".to_owned())
            })?;
            header
                .validate()
                .map_err(|msg| KvError::InvalidDimensions(format!("invalid ArenaHeader: {msg}")))?;
        }
        Ok(Self { arena })
    }

    /// Returns the validated [`ArenaHeader`] at the base of the mapped arena.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if the header fails validation.
    pub fn header(&self) -> Result<&ArenaHeader, KvError> {
        let (header, _) = ArenaHeader::ref_from_prefix(self.arena.as_slice()).map_err(|_| {
            KvError::InvalidDimensions("arena buffer smaller than ArenaHeader".to_owned())
        })?;
        header
            .validate()
            .map_err(|msg| KvError::InvalidDimensions(format!("invalid ArenaHeader: {msg}")))?;
        Ok(header)
    }

    /// Returns a contiguous layer-major `[layer][K|V][pos][kv_dim]` mutable [`KvCacheView`] over
    /// the arena's initialized KV slab, bounded strictly to `seq_capacity` without trailing
    /// cacheline-padding drift.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if header validation fails, `kv_dtype != F32`,
    /// `active_slots != 1`, MLA latent compression is active, dimensions overflow `usize`, or
    /// the KV slab slice is unaligned or out of bounds.
    pub fn kv_cache_view_mut(&mut self) -> Result<KvCacheView<'_>, KvError> {
        kv_cache_view_from_arena_slice_mut(self.arena.as_mut_slice())
    }

    /// Returns a zero-copy [`ArenaKvWriter`] over the arena's KV slab and `KvBlockMeta` table.
    ///
    /// # Errors
    /// Returns [`KvError::IpcError`] if the arena mapping is read-only or unaligned for atomics,
    /// or [`KvError::InvalidDimensions`] if header, KV slab, or block-meta geometry checks fail.
    pub fn kv_writer(&mut self) -> Result<ArenaKvWriter<'_>, KvError> {
        ArenaKvWriter::from_arena(self.arena)
    }

    /// Returns a zero-copy read-only [`ArenaKvReader`] over the arena's KV slab and `KvBlockMeta` table.
    ///
    /// # Errors
    /// Returns [`KvError::IpcError`] if the metadata table is unaligned for atomics, or
    /// [`KvError::InvalidDimensions`] if header, KV slab, or block-meta geometry checks fail.
    pub fn kv_reader(&self) -> Result<ArenaKvReader<'_>, KvError> {
        ArenaKvReader::from_arena(self.arena)
    }

    /// Advances [`KvBlockMeta::committed_token_count`] monotonically across all 16-token blocks
    /// spanned by `[start_pos .. start_pos + seq_len)` using atomic `Release` stores.
    ///
    /// # Errors
    /// Returns [`KvError`] if the span is non-contiguous with the current committed watermark,
    /// exceeds `seq_capacity`, overflows `usize`, or the arena is read-only or malformed.
    pub fn commit_token_span(&mut self, start_pos: usize, seq_len: usize) -> Result<(), KvError> {
        self.kv_writer()?.commit_token_span(start_pos, seq_len)
    }

    /// Alias for [`Self::commit_token_span`] committing a batched prefill token span across
    /// spanned 16-token blocks with `Release` ordering.
    ///
    /// # Errors
    /// Returns [`KvError`] if [`Self::commit_token_span`] fails.
    pub fn commit_prefill_range(
        &mut self,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(), KvError> {
        self.commit_token_span(start_pos, seq_len)
    }

    /// Returns the total number of contiguously committed tokens across this arena's `KvBlockMeta`
    /// table, observed with `Acquire` ordering.
    ///
    /// # Errors
    /// Returns [`KvError`] if header validation fails or block metadata is non-contiguous.
    pub fn committed_tokens(&self) -> Result<usize, KvError> {
        self.kv_reader()?.committed_tokens()
    }

    /// Reads and validates the active [`RecurrentStateSlotHeader`] for an `O(1)` recurrent or
    /// hybrid arena.
    ///
    /// # Errors
    /// Returns [`KvError`] if header validation fails, the arena has no recurrent state region,
    /// or the slot header is invalid.
    pub fn recurrent_header(&self) -> Result<RecurrentStateSlotHeader, KvError> {
        let arena_hdr = self.header()?;
        let rec_off = usize::try_from(recurrent_region_offset(arena_hdr)).map_err(|_| {
            KvError::InvalidDimensions("recurrent_region_offset exceeds address space".to_owned())
        })?;
        let region = self.arena.as_slice().get(rec_off..).ok_or_else(|| {
            KvError::IpcError("arena buffer missing recurrent state region".to_owned())
        })?;
        read_recurrent_slot_header_from_region(region)
    }

    /// Returns `(header, active_bank_readonly, inactive_bank_mutable)` for zero-tearing
    /// ping-pong recurrent execution over this arena.
    ///
    /// # Errors
    /// Returns [`KvError::IpcError`] if the arena is read-only or truncated, or
    /// [`KvError::InvalidDimensions`] if header validation fails.
    pub fn recurrent_banks_mut(
        &mut self,
    ) -> Result<(RecurrentStateSlotHeader, &[u8], &mut [u8]), KvError> {
        if self.arena.is_read_only() {
            return Err(KvError::IpcError(
                "cannot obtain mutable recurrent banks from read-only SharedMemoryArena".to_owned(),
            ));
        }
        recurrent_banks_in_arena_mut(self.arena.as_mut_slice())
    }

    /// Atomically publishes the inactive recurrent staging bank as the active bank with
    /// `Release` ordering (`O(1)` ping-pong commit).
    ///
    /// # Errors
    /// Returns [`KvError::IpcError`] if the arena is read-only or truncated, or
    /// [`KvError::InvalidDimensions`] if `new_step` is non-monotonic or header validation fails.
    pub fn commit_recurrent_step(
        &mut self,
        new_step: u32,
    ) -> Result<RecurrentStateSlotHeader, KvError> {
        if self.arena.is_read_only() {
            return Err(KvError::IpcError(
                "cannot commit recurrent step on read-only SharedMemoryArena".to_owned(),
            ));
        }
        commit_recurrent_step_in_arena(self.arena.as_mut_slice(), new_step)
    }

    /// Consumes the handle and returns the underlying [`SharedMemoryArena`].
    #[must_use]
    pub fn into_inner(self) -> &'a mut SharedMemoryArena {
        self.arena
    }
}

const COMMITTED_TOKEN_COUNT_OFFSET: usize = offset_of!(KvBlockMeta, committed_token_count);
const _: () = assert!(KV_BLOCK_TOKEN_CAPACITY == 16);

fn align_up_usize_checked(value: usize, align: usize) -> Result<usize, KvError> {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    value
        .checked_add(mask)
        .map(|aligned| aligned & !mask)
        .ok_or_else(|| {
            KvError::InvalidDimensions("aligned byte length overflowed usize".to_owned())
        })
}

#[derive(Debug, Clone, Copy)]
struct KvSlabGeometry {
    geometry: KvGeometry,
    kv_dtype: KvDtype,
    kv_dim: usize,
    kv_row_bytes: usize,
    raw_plane_bytes: usize,
    kv_plane_stride_bytes: usize,
}

impl KvSlabGeometry {
    const fn plane_len(self) -> usize {
        self.geometry.seq_capacity * self.kv_dim
    }

    const fn valid_start_pos(self, end_pos: usize) -> usize {
        match self.geometry.sliding_window {
            Some(window) => end_pos.saturating_sub(window),
            None => 0,
        }
    }

    fn layer_plane_bounds(
        self,
        layer: usize,
    ) -> Result<(core::ops::Range<usize>, core::ops::Range<usize>), KvError> {
        if layer >= self.geometry.num_layers {
            return Err(KvError::InvalidDimensions(format!(
                "layer index {layer} out of bounds for num_layers={}",
                self.geometry.num_layers
            )));
        }
        let layer_stride = self.kv_plane_stride_bytes.checked_mul(2).ok_or_else(|| {
            KvError::InvalidDimensions("KV layer byte stride overflowed usize".to_owned())
        })?;
        let k_start = layer.checked_mul(layer_stride).ok_or_else(|| {
            KvError::InvalidDimensions("KV layer offset overflowed usize".to_owned())
        })?;
        let k_end = k_start.checked_add(self.raw_plane_bytes).ok_or_else(|| {
            KvError::InvalidDimensions("KV K-plane end offset overflowed usize".to_owned())
        })?;
        let v_start = k_start
            .checked_add(self.kv_plane_stride_bytes)
            .ok_or_else(|| {
                KvError::InvalidDimensions("KV V-plane offset overflowed usize".to_owned())
            })?;
        let v_end = v_start.checked_add(self.raw_plane_bytes).ok_or_else(|| {
            KvError::InvalidDimensions("KV V-plane end offset overflowed usize".to_owned())
        })?;
        Ok((k_start..k_end, v_start..v_end))
    }

    fn validate_pos_range(
        self,
        start_pos: usize,
        end_pos: usize,
    ) -> Result<(usize, usize), KvError> {
        if start_pos > end_pos {
            return Err(KvError::InvalidDimensions(format!(
                "start_pos ({start_pos}) > end_pos ({end_pos})"
            )));
        }
        if end_pos > self.geometry.seq_capacity {
            return Err(KvError::InvalidDimensions(format!(
                "end_pos ({end_pos}) exceeds seq_capacity ({})",
                self.geometry.seq_capacity
            )));
        }
        let seq_len = end_pos - start_pos;
        let elem_len = seq_len.checked_mul(self.kv_dim).ok_or_else(|| {
            KvError::InvalidDimensions("KV element range length overflowed usize".to_owned())
        })?;
        Ok((seq_len, elem_len))
    }

    fn validate_batch_write(
        self,
        layer: usize,
        start_pos: usize,
        k_len: usize,
        v_len: usize,
    ) -> Result<(usize, usize), KvError> {
        if layer >= self.geometry.num_layers {
            return Err(KvError::InvalidDimensions(format!(
                "layer index {layer} out of bounds for num_layers={}",
                self.geometry.num_layers
            )));
        }
        if k_len != v_len {
            return Err(KvError::InvalidDimensions(format!(
                "k_batch len ({k_len}) does not match v_batch len ({v_len})"
            )));
        }
        if k_len % self.kv_dim != 0 {
            return Err(KvError::InvalidDimensions(format!(
                "k_batch len ({k_len}) is not a multiple of kv_dim ({})",
                self.kv_dim
            )));
        }
        let seq_len = k_len / self.kv_dim;
        let end_pos = start_pos.checked_add(seq_len).ok_or_else(|| {
            KvError::InvalidDimensions("append batch end_pos overflowed usize".to_owned())
        })?;
        if end_pos > self.geometry.seq_capacity {
            return Err(KvError::InvalidDimensions(format!(
                "batch span [{start_pos}..{end_pos}) exceeds seq_capacity ({})",
                self.geometry.seq_capacity
            )));
        }
        Ok((seq_len, end_pos))
    }
}

fn read_kv_bytes_planes(
    kv_bytes: &[u8],
    slab: KvSlabGeometry,
    layer: usize,
) -> Result<(&[u8], &[u8]), KvError> {
    let (k_range, v_range) = slab.layer_plane_bounds(layer)?;
    let k_plane = kv_bytes.get(k_range).ok_or_else(|| {
        KvError::InvalidDimensions("KV slab K-plane range exceeds buffer bounds".to_owned())
    })?;
    let v_plane = kv_bytes.get(v_range).ok_or_else(|| {
        KvError::InvalidDimensions("KV slab V-plane range exceeds buffer bounds".to_owned())
    })?;
    Ok((k_plane, v_plane))
}

fn read_kv_f32_planes(
    kv_bytes: &[u8],
    slab: KvSlabGeometry,
    layer: usize,
) -> Result<(&[f32], &[f32]), KvError> {
    if slab.kv_dtype != KvDtype::F32 {
        return Err(KvError::InvalidDimensions(format!(
            "contiguous &[f32] KV plane view requires F32 dtype, got {:?}",
            slab.kv_dtype
        )));
    }
    let (k_plane, v_plane) = read_kv_bytes_planes(kv_bytes, slab, layer)?;
    let k_f32 = <[f32]>::ref_from_bytes(k_plane).map_err(|err| {
        KvError::InvalidDimensions(format!("KV K-plane f32 conversion failed: {err:?}"))
    })?;
    let v_f32 = <[f32]>::ref_from_bytes(v_plane).map_err(|err| {
        KvError::InvalidDimensions(format!("KV V-plane f32 conversion failed: {err:?}"))
    })?;
    Ok((k_f32, v_f32))
}

fn read_kv_f32_range(
    kv_bytes: &[u8],
    slab: KvSlabGeometry,
    layer: usize,
    start_pos: usize,
    end_pos: usize,
) -> Result<(&[f32], &[f32]), KvError> {
    let (_seq_len, _elem_len) = slab.validate_pos_range(start_pos, end_pos)?;
    let (k_plane, v_plane) = read_kv_f32_planes(kv_bytes, slab, layer)?;
    let elem_start = start_pos
        .checked_mul(slab.kv_dim)
        .ok_or_else(|| KvError::InvalidDimensions("elem_start overflowed usize".to_owned()))?;
    let elem_end = end_pos
        .checked_mul(slab.kv_dim)
        .ok_or_else(|| KvError::InvalidDimensions("elem_end overflowed usize".to_owned()))?;
    Ok((
        &k_plane[elem_start..elem_end],
        &v_plane[elem_start..elem_end],
    ))
}

fn write_kv_f32_planes<'a>(
    kv_bytes: &'a mut [u8],
    slab: KvSlabGeometry,
    layer: usize,
    start_pos: usize,
    k_batch: &[f32],
    v_batch: &[f32],
) -> Result<(&'a [f32], &'a [f32]), KvError> {
    if slab.kv_dtype != KvDtype::F32 {
        return Err(KvError::InvalidDimensions(format!(
            "append_layer_batch requires F32 KV cache, got {:?}; use append_quantized_kv_batch",
            slab.kv_dtype
        )));
    }
    let (_seq_len, end_pos) =
        slab.validate_batch_write(layer, start_pos, k_batch.len(), v_batch.len())?;
    let (k_range, v_range) = slab.layer_plane_bounds(layer)?;
    if v_range.end > kv_bytes.len() {
        return Err(KvError::InvalidDimensions(
            "KV layer plane range exceeds slab byte length".to_owned(),
        ));
    }
    let elem_start = start_pos
        .checked_mul(slab.kv_dim)
        .ok_or_else(|| KvError::InvalidDimensions("elem_start overflowed usize".to_owned()))?;
    let elem_end = end_pos
        .checked_mul(slab.kv_dim)
        .ok_or_else(|| KvError::InvalidDimensions("elem_end overflowed usize".to_owned()))?;
    let (prefix, tail) = kv_bytes.split_at_mut(v_range.start);
    let k_plane = <[f32]>::mut_from_bytes(&mut prefix[k_range]).map_err(|err| {
        KvError::InvalidDimensions(format!("KV K-plane f32 conversion failed: {err:?}"))
    })?;
    let v_plane = <[f32]>::mut_from_bytes(&mut tail[..v_range.len()]).map_err(|err| {
        KvError::InvalidDimensions(format!("KV V-plane f32 conversion failed: {err:?}"))
    })?;
    k_plane[elem_start..elem_end].copy_from_slice(k_batch);
    v_plane[elem_start..elem_end].copy_from_slice(v_batch);
    Ok((&k_plane[..elem_end], &v_plane[..elem_end]))
}

fn unpack_kv_planes_range_f32<'o>(
    slab: KvSlabGeometry,
    k_plane: &[u8],
    v_plane: &[u8],
    pos_range: core::ops::Range<usize>,
    k_out: &'o mut [f32],
    v_out: &'o mut [f32],
) -> Result<(&'o [f32], &'o [f32]), KvError> {
    let k_written = unpack_kv_range_plane(
        slab.kv_dtype,
        slab.kv_dim,
        pos_range.start,
        pos_range.end,
        k_plane,
        k_out,
    )?;
    let v_written = unpack_kv_range_plane(
        slab.kv_dtype,
        slab.kv_dim,
        pos_range.start,
        pos_range.end,
        v_plane,
        v_out,
    )?;
    debug_assert_eq!(k_written, v_written);
    Ok((&k_out[..k_written], &v_out[..v_written]))
}

fn alloc_scratch_kv_pair<'s>(
    scope: &'s ScratchScope<'_>,
    total_elems: usize,
) -> Result<(&'s mut [f32], &'s mut [f32]), KvError> {
    if total_elems == 0 {
        return Ok((&mut [], &mut []));
    }
    let k_buf = scope
        .alloc_f32(total_elems)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let v_buf = scope
        .alloc_f32(total_elems)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    Ok((k_buf, v_buf))
}

#[derive(Debug, Clone, Copy)]
struct ValidatedArenaKvSlab {
    slab: KvSlabGeometry,
    kv_offset: usize,
    kv_end: usize,
}

fn extract_slab_geometry(header: &ArenaHeader) -> Result<KvSlabGeometry, KvError> {
    let kv_dtype = header
        .kv_dtype()
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    if header._reserved[ARENA_RESERVED_MLA_FLAG_OFFSET] != 0 {
        return Err(KvError::InvalidDimensions(
            "KV slab view/reader does not support MLA latent-compressed KV slabs".to_owned(),
        ));
    }
    if header.num_layers == 0 || header.head_count_kv == 0 || header.head_dim == 0 {
        return Err(KvError::InvalidDimensions(
            "ArenaHeader KV geometry dimensions must be greater than zero".to_owned(),
        ));
    }
    let active_slots = header.active_slots.max(1);
    if active_slots != 1 {
        return Err(KvError::InvalidDimensions(format!(
            "KvCacheView expects single active slot, arena has active_slots={active_slots}"
        )));
    }
    let seq_cap_u32 = arena_seq_capacity(header);
    if seq_cap_u32 == 0 {
        return Err(KvError::InvalidDimensions(
            "ArenaHeader seq_capacity must be greater than zero".to_owned(),
        ));
    }

    let num_layers = usize::try_from(header.num_layers)
        .map_err(|_| KvError::InvalidDimensions("num_layers overflowed usize".to_owned()))?;
    let n_heads_kv = usize::try_from(header.head_count_kv)
        .map_err(|_| KvError::InvalidDimensions("head_count_kv overflowed usize".to_owned()))?;
    let head_dim = usize::try_from(header.head_dim)
        .map_err(|_| KvError::InvalidDimensions("head_dim overflowed usize".to_owned()))?;
    let seq_capacity = usize::try_from(seq_cap_u32)
        .map_err(|_| KvError::InvalidDimensions("seq_capacity overflowed usize".to_owned()))?;
    let sliding_window = if header.swa_window_size > 0 {
        Some(usize::try_from(header.swa_window_size).map_err(|_| {
            KvError::InvalidDimensions("swa_window_size overflowed usize".to_owned())
        })?)
    } else {
        None
    };

    let geometry = KvGeometry::new(
        num_layers,
        n_heads_kv,
        head_dim,
        seq_capacity,
        sliding_window,
    )
    .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let kv_dim = geometry
        .kv_dim()
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let kv_row_bytes = geometry
        .row_bytes(kv_dtype)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let raw_plane_bytes = geometry
        .plane_bytes(kv_dtype)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
    let kv_plane_stride_bytes = align_up_usize_checked(raw_plane_bytes, VECTOR_PLANE_ALIGNMENT)?;

    Ok(KvSlabGeometry {
        geometry,
        kv_dtype,
        kv_dim,
        kv_row_bytes,
        raw_plane_bytes,
        kv_plane_stride_bytes,
    })
}

fn validate_arena_kv_slab(arena_bytes: &[u8]) -> Result<ValidatedArenaKvSlab, KvError> {
    let (header, _) = ArenaHeader::ref_from_prefix(arena_bytes).map_err(|_| {
        KvError::InvalidDimensions("arena buffer smaller than ArenaHeader".to_owned())
    })?;
    header
        .validate()
        .map_err(|msg| KvError::InvalidDimensions(format!("invalid ArenaHeader: {msg}")))?;

    let slab = extract_slab_geometry(header)?;
    let kv_offset = usize::try_from(header.kv_blocks_offset)
        .map_err(|_| KvError::InvalidDimensions("kv_blocks_offset overflowed usize".to_owned()))?;
    if kv_offset % VECTOR_PLANE_ALIGNMENT != 0 {
        return Err(KvError::InvalidDimensions(format!(
            "kv_blocks_offset ({kv_offset}) must be {VECTOR_PLANE_ALIGNMENT}-byte aligned"
        )));
    }
    let output_offset = usize::try_from(header.output_slabs_offset).map_err(|_| {
        KvError::InvalidDimensions("output_slabs_offset overflowed usize".to_owned())
    })?;
    let total_bytes = usize::try_from(header.total_bytes)
        .map_err(|_| KvError::InvalidDimensions("total_bytes overflowed usize".to_owned()))?;

    let total_planes =
        slab.geometry.num_layers.checked_mul(2).ok_or_else(|| {
            KvError::InvalidDimensions("KV plane count overflowed usize".to_owned())
        })?;
    let required_bytes = slab
        .kv_plane_stride_bytes
        .checked_mul(total_planes)
        .ok_or_else(|| {
            KvError::InvalidDimensions("KV slab byte length overflowed usize".to_owned())
        })?;
    let kv_end = kv_offset.checked_add(required_bytes).ok_or_else(|| {
        KvError::InvalidDimensions("KV slab end offset overflowed usize".to_owned())
    })?;

    if kv_offset < size_of::<ArenaHeader>()
        || output_offset < kv_end
        || total_bytes > arena_bytes.len()
        || kv_end > arena_bytes.len()
    {
        return Err(KvError::InvalidDimensions(format!(
            "KV slab bounds [{kv_offset}..{kv_end}) invalid for output_slabs_offset={output_offset}, arena_len={}",
            arena_bytes.len()
        )));
    }

    Ok(ValidatedArenaKvSlab {
        slab,
        kv_offset,
        kv_end,
    })
}

fn validate_arena_meta_bounds(
    arena_bytes: &[u8],
    seq_capacity: usize,
    kv_offset: usize,
) -> Result<(usize, usize, usize), KvError> {
    if seq_capacity % KV_BLOCK_TOKEN_CAPACITY != 0 {
        return Err(KvError::InvalidDimensions(format!(
            "seq_capacity ({seq_capacity}) must be a multiple of KV_BLOCK_TOKEN_CAPACITY ({KV_BLOCK_TOKEN_CAPACITY})"
        )));
    }
    let logical_block_count = seq_capacity / KV_BLOCK_TOKEN_CAPACITY;
    let (header, _) = ArenaHeader::ref_from_prefix(arena_bytes).map_err(|_| {
        KvError::InvalidDimensions("arena buffer smaller than ArenaHeader".to_owned())
    })?;
    let header_block_count = usize::try_from(header.block_count)
        .map_err(|_| KvError::InvalidDimensions("block_count overflowed usize".to_owned()))?;
    if header_block_count < logical_block_count {
        return Err(KvError::InvalidDimensions(format!(
            "header.block_count ({header_block_count}) smaller than required logical blocks ({logical_block_count})"
        )));
    }

    let meta_offset = usize::try_from(header.meta_table_offset)
        .map_err(|_| KvError::InvalidDimensions("meta_table_offset overflowed usize".to_owned()))?;
    let meta_size = header_block_count
        .checked_mul(size_of::<KvBlockMeta>())
        .ok_or_else(|| {
            KvError::InvalidDimensions("meta_table byte size overflowed usize".to_owned())
        })?;
    let meta_end = meta_offset
        .checked_add(meta_size)
        .ok_or_else(|| KvError::InvalidDimensions("meta_table end overflowed usize".to_owned()))?;
    if meta_offset < size_of::<ArenaHeader>()
        || meta_end > kv_offset
        || meta_end > arena_bytes.len()
    {
        return Err(KvError::InvalidDimensions(format!(
            "meta_table bounds [{meta_offset}..{meta_end}) overlap or exceed arena bounds (kv_offset={kv_offset}, arena_len={})",
            arena_bytes.len()
        )));
    }
    Ok((meta_offset, meta_end, logical_block_count))
}

fn validate_meta_slice_for_atomic(
    meta_bytes: &[u8],
    logical_block_count: usize,
) -> Result<(), KvError> {
    let required = logical_block_count
        .checked_mul(size_of::<KvBlockMeta>())
        .ok_or_else(|| {
            KvError::InvalidDimensions(
                "logical_block_count byte length overflowed usize".to_owned(),
            )
        })?;
    if meta_bytes.len() < required {
        return Err(KvError::InvalidDimensions(format!(
            "meta_table byte len ({}) smaller than required ({required})",
            meta_bytes.len()
        )));
    }
    let field_addr = (meta_bytes.as_ptr() as usize) + COMMITTED_TOKEN_COUNT_OFFSET;
    if field_addr % align_of::<AtomicU32>() != 0
        || size_of::<KvBlockMeta>() % align_of::<AtomicU32>() != 0
    {
        return Err(KvError::IpcError(
            "KvBlockMeta committed_token_count field is not 4-byte aligned for AtomicU32 access"
                .to_owned(),
        ));
    }
    Ok(())
}

fn block_committed_field_offset(
    block_idx: usize,
    logical_block_count: usize,
    meta_len: usize,
) -> Result<usize, KvError> {
    if block_idx >= logical_block_count {
        return Err(KvError::InvalidDimensions(format!(
            "block index {block_idx} out of bounds for logical_block_count={logical_block_count}"
        )));
    }
    let entry_start = block_idx * size_of::<KvBlockMeta>();
    let entry_end = entry_start + size_of::<KvBlockMeta>();
    if entry_end > meta_len {
        return Err(KvError::InvalidDimensions(format!(
            "block metadata entry[{block_idx}] end ({entry_end}) exceeds table len ({meta_len})"
        )));
    }
    Ok(entry_start + COMMITTED_TOKEN_COUNT_OFFSET)
}

fn read_block_committed_tokens_from_slice(
    meta_bytes: &[u8],
    block_idx: usize,
    logical_block_count: usize,
) -> Result<u32, KvError> {
    let field_off = block_committed_field_offset(block_idx, logical_block_count, meta_bytes.len())?;
    let field_slice = &meta_bytes[field_off..field_off + size_of::<u32>()];
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "4-byte AtomicU32 alignment is verified by validate_meta_slice_for_atomic"
    )]
    // SAFETY: `field_slice` is 4 bytes within `meta_bytes` and 4-byte aligned as verified by
    // `validate_meta_slice_for_atomic`; a single aligned `u32` `read_volatile` avoids multi-byte
    // torn loads while preserving `&[u8]` shared-read provenance, and the trailing `Acquire` fence
    // orders subsequent KV slab reads after the observed watermark.
    let val = unsafe { field_slice.as_ptr().cast::<u32>().read_volatile() };
    fence(Ordering::Acquire);
    Ok(val)
}

fn store_block_committed_tokens_in_slice(
    meta_bytes: &mut [u8],
    block_idx: usize,
    logical_block_count: usize,
    new_count: u32,
) -> Result<(), KvError> {
    let field_off = block_committed_field_offset(block_idx, logical_block_count, meta_bytes.len())?;
    let field_slice = &mut meta_bytes[field_off..field_off + size_of::<u32>()];
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "4-byte AtomicU32 alignment is verified by validate_meta_slice_for_atomic"
    )]
    // SAFETY: `field_slice` is exclusively borrowed (`&mut [u8]`), 4 bytes long, and 4-byte
    // (`AtomicU32`) aligned as verified by `validate_meta_slice_for_atomic`.
    unsafe {
        AtomicU32::from_ptr(field_slice.as_mut_ptr().cast::<u32>())
            .store(new_count, Ordering::Release);
    }
    Ok(())
}

fn committed_watermark_from_meta(
    meta_bytes: &[u8],
    logical_block_count: usize,
) -> Result<usize, KvError> {
    let mut watermark = 0_usize;
    let mut seen_incomplete = false;
    for block_idx in 0..logical_block_count {
        let count_u32 =
            read_block_committed_tokens_from_slice(meta_bytes, block_idx, logical_block_count)?;
        let count = usize::try_from(count_u32).map_err(|_| {
            KvError::InvalidDimensions("committed_token_count overflowed usize".to_owned())
        })?;
        if count > KV_BLOCK_TOKEN_CAPACITY {
            return Err(KvError::InvalidDimensions(format!(
                "block[{block_idx}] committed_token_count ({count}) exceeds capacity ({KV_BLOCK_TOKEN_CAPACITY})"
            )));
        }
        if seen_incomplete && count > 0 {
            return Err(KvError::InvalidDimensions(format!(
                "non-contiguous committed_token_count in block[{block_idx}] ({count}) after incomplete prior block"
            )));
        }
        watermark = watermark.checked_add(count).ok_or_else(|| {
            KvError::InvalidDimensions("committed token watermark overflowed usize".to_owned())
        })?;
        if count < KV_BLOCK_TOKEN_CAPACITY {
            seen_incomplete = true;
        }
    }
    Ok(watermark)
}

fn commit_token_span_in_meta(
    meta_bytes: &mut [u8],
    logical_block_count: usize,
    seq_capacity: usize,
    start_pos: usize,
    seq_len: usize,
) -> Result<(), KvError> {
    let end_pos = start_pos.checked_add(seq_len).ok_or_else(|| {
        KvError::InvalidDimensions("commit_token_span end_pos overflowed usize".to_owned())
    })?;
    if end_pos > seq_capacity {
        return Err(KvError::InvalidDimensions(format!(
            "commit span [{start_pos}..{end_pos}) exceeds seq_capacity ({seq_capacity})"
        )));
    }
    let watermark = committed_watermark_from_meta(meta_bytes, logical_block_count)?;
    if start_pos != watermark {
        return Err(KvError::InvalidDimensions(format!(
            "non-contiguous commit span: start_pos ({start_pos}) != current committed watermark ({watermark})"
        )));
    }
    if seq_len == 0 {
        return Ok(());
    }

    let first_block = start_pos / KV_BLOCK_TOKEN_CAPACITY;
    let last_block = (end_pos - 1) / KV_BLOCK_TOKEN_CAPACITY;
    if last_block >= logical_block_count {
        return Err(KvError::InvalidDimensions(format!(
            "last spanned block ({last_block}) exceeds logical_block_count ({logical_block_count})"
        )));
    }

    // Ensure prior KV plane writes happen-before consumers observe advanced committed counts.
    fence(Ordering::Release);
    for block_idx in first_block..=last_block {
        let block_start = block_idx * KV_BLOCK_TOKEN_CAPACITY;
        let new_count_usize = (end_pos - block_start).min(KV_BLOCK_TOKEN_CAPACITY);
        let new_count = u32::try_from(new_count_usize).map_err(|_| {
            KvError::InvalidDimensions("block committed_token_count overflowed u32".to_owned())
        })?;
        store_block_committed_tokens_in_slice(
            meta_bytes,
            block_idx,
            logical_block_count,
            new_count,
        )?;
    }

    Ok(())
}

fn unpack_kv_planes_in_scratch<'s>(
    slab: KvSlabGeometry,
    k_plane: &[u8],
    v_plane: &[u8],
    start_pos: usize,
    end_pos: usize,
    scope: &'s ScratchScope<'_>,
) -> Result<(&'s [f32], &'s [f32]), KvError> {
    let (_seq_len, elem_len) = slab.validate_pos_range(start_pos, end_pos)?;
    let (k_buf, v_buf) = alloc_scratch_kv_pair(scope, elem_len)?;
    unpack_kv_planes_range_f32(slab, k_plane, v_plane, start_pos..end_pos, k_buf, v_buf)
}

#[derive(Debug)]
enum WriterBacking<'a> {
    F32View(KvCacheView<'a>),
    Bytes {
        kv_bytes: &'a mut [u8],
        slab: KvSlabGeometry,
    },
}

/// Zero-copy batched KV slab writer and atomic [`KvBlockMeta`] committer over an initialized arena.
#[derive(Debug)]
pub struct ArenaKvWriter<'a> {
    backing: WriterBacking<'a>,
    meta_bytes: &'a mut [u8],
    logical_block_count: usize,
}

impl<'a> ArenaKvWriter<'a> {
    fn slab(&self) -> KvSlabGeometry {
        match &self.backing {
            WriterBacking::F32View(view) => {
                let geometry = view.geometry();
                let kv_dim = view.kv_dim();
                let kv_row_bytes = kv_dim * size_of::<f32>();
                let raw_plane_bytes = geometry.seq_capacity * kv_row_bytes;
                KvSlabGeometry {
                    geometry,
                    kv_dtype: KvDtype::F32,
                    kv_dim,
                    kv_row_bytes,
                    raw_plane_bytes,
                    kv_plane_stride_bytes: raw_plane_bytes,
                }
            }
            WriterBacking::Bytes { slab, .. } => *slab,
        }
    }

    /// Wraps an initialized shared memory arena byte slice in a bounds-checked [`ArenaKvWriter`].
    ///
    /// Supports `F32`, `F16`, and `Q8_0` KV slabs aligned to [`VECTOR_PLANE_ALIGNMENT`].
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if header, KV slab, or `KvBlockMeta` bounds fail, or
    /// [`KvError::IpcError`] if alignment is invalid for `AtomicU32` metadata commits.
    pub fn from_arena_slice(arena_bytes: &'a mut [u8]) -> Result<Self, KvError> {
        let validated = validate_arena_kv_slab(arena_bytes)?;
        let (meta_offset, meta_end, logical_block_count) = validate_arena_meta_bounds(
            arena_bytes,
            validated.slab.geometry.seq_capacity,
            validated.kv_offset,
        )?;
        let (head, tail) = arena_bytes.split_at_mut(validated.kv_offset);
        let meta_bytes = &mut head[meta_offset..meta_end];
        validate_meta_slice_for_atomic(meta_bytes, logical_block_count)?;

        let kv_bytes = &mut tail[..(validated.kv_end - validated.kv_offset)];
        Ok(Self {
            backing: WriterBacking::Bytes {
                kv_bytes,
                slab: validated.slab,
            },
            meta_bytes,
            logical_block_count,
        })
    }

    /// Wraps a writable [`SharedMemoryArena`] in a bounds-checked [`ArenaKvWriter`].
    ///
    /// # Errors
    /// Returns [`KvError::IpcError`] if `arena` is read-only or unaligned, or
    /// [`KvError::InvalidDimensions`] if header or layout checks fail.
    pub fn from_arena(arena: &'a mut SharedMemoryArena) -> Result<Self, KvError> {
        if arena.is_read_only() {
            return Err(KvError::IpcError(
                "cannot construct ArenaKvWriter from read-only SharedMemoryArena".to_owned(),
            ));
        }
        Self::from_arena_slice(arena.as_mut_slice())
    }

    /// Constructs an [`ArenaKvWriter`] directly from a [`KvCacheView`] and its backing
    /// [`KvBlockMeta`] slice.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `seq_capacity` is not a multiple of
    /// [`KV_BLOCK_TOKEN_CAPACITY`] or `block_meta` has fewer entries than sequence blocks.
    pub fn from_parts(
        kv_view: KvCacheView<'a>,
        block_meta: &'a mut [KvBlockMeta],
    ) -> Result<Self, KvError> {
        let seq_capacity = kv_view.seq_capacity();
        if seq_capacity % KV_BLOCK_TOKEN_CAPACITY != 0 {
            return Err(KvError::InvalidDimensions(format!(
                "seq_capacity ({seq_capacity}) must be a multiple of KV_BLOCK_TOKEN_CAPACITY ({KV_BLOCK_TOKEN_CAPACITY})"
            )));
        }
        let logical_block_count = seq_capacity / KV_BLOCK_TOKEN_CAPACITY;
        if block_meta.len() < logical_block_count {
            return Err(KvError::InvalidDimensions(format!(
                "block_meta len ({}) smaller than required logical blocks ({logical_block_count})",
                block_meta.len()
            )));
        }
        let meta_bytes = block_meta.as_mut_bytes();
        validate_meta_slice_for_atomic(meta_bytes, logical_block_count)?;
        Ok(Self {
            backing: WriterBacking::F32View(kv_view),
            meta_bytes,
            logical_block_count,
        })
    }

    /// Returns the validated [`KvGeometry`].
    #[must_use]
    pub const fn geometry(&self) -> KvGeometry {
        match &self.backing {
            WriterBacking::F32View(view) => view.geometry(),
            WriterBacking::Bytes { slab, .. } => slab.geometry,
        }
    }

    /// Returns the maximum sequence token capacity (`seq_capacity`).
    #[must_use]
    pub const fn seq_capacity(&self) -> usize {
        self.geometry().seq_capacity
    }

    /// Returns the per-position vector width (`n_heads_kv * head_dim`).
    #[must_use]
    pub const fn kv_dim(&self) -> usize {
        match &self.backing {
            WriterBacking::F32View(view) => view.kv_dim(),
            WriterBacking::Bytes { slab, .. } => slab.kv_dim,
        }
    }

    /// Returns the element encoding ([`KvDtype`]) of the underlying KV slab.
    #[must_use]
    pub const fn kv_dtype(&self) -> KvDtype {
        match &self.backing {
            WriterBacking::F32View(_) => KvDtype::F32,
            WriterBacking::Bytes { slab, .. } => slab.kv_dtype,
        }
    }

    /// Returns the encoded byte stride per token row (`KvDtype::row_bytes(kv_dim)`).
    #[must_use]
    pub const fn kv_row_bytes(&self) -> usize {
        match &self.backing {
            WriterBacking::F32View(view) => view.kv_dim() * size_of::<f32>(),
            WriterBacking::Bytes { slab, .. } => slab.kv_row_bytes,
        }
    }

    /// Returns the aligned byte stride between consecutive K/V planes.
    #[must_use]
    pub const fn kv_plane_stride_bytes(&self) -> usize {
        match &self.backing {
            WriterBacking::F32View(view) => {
                view.geometry().seq_capacity * view.kv_dim() * size_of::<f32>()
            }
            WriterBacking::Bytes { slab, .. } => slab.kv_plane_stride_bytes,
        }
    }

    /// Returns the number of 16-token blocks backing `seq_capacity`.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.logical_block_count
    }

    /// Returns shared access to the underlying [`KvCacheView`] when constructed via [`Self::from_parts`].
    #[must_use]
    pub const fn kv_cache_view(&self) -> Option<&KvCacheView<'a>> {
        match &self.backing {
            WriterBacking::F32View(view) => Some(view),
            WriterBacking::Bytes { .. } => None,
        }
    }

    /// Returns mutable access to the underlying [`KvCacheView`] when constructed via [`Self::from_parts`].
    #[must_use]
    pub fn kv_cache_view_mut(&mut self) -> Option<&mut KvCacheView<'a>> {
        match &mut self.backing {
            WriterBacking::F32View(view) => Some(view),
            WriterBacking::Bytes { .. } => None,
        }
    }

    /// Copies contiguous `[seq_len, kv_dim]` `F32` K and V batch slices for `layer` starting at
    /// `start_pos` in a single copy per plane, returning contiguous `0..(start_pos + seq_len)`
    /// `(k_prefix, v_prefix)` reader slices.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `kv_dtype != F32`, `layer >= num_layers`,
    /// `k_batch.len() != v_batch.len()`, slice lengths are not multiples of `kv_dim`, or
    /// `start_pos + seq_len > seq_capacity`.
    pub fn append_layer_batch(
        &mut self,
        layer: usize,
        start_pos: usize,
        k_batch: &[f32],
        v_batch: &[f32],
    ) -> Result<(&[f32], &[f32]), KvError> {
        let slab = self.slab();
        match &mut self.backing {
            WriterBacking::F32View(view) => {
                let (_seq_len, end_pos) =
                    slab.validate_batch_write(layer, start_pos, k_batch.len(), v_batch.len())?;
                let elem_start = start_pos.checked_mul(slab.kv_dim).ok_or_else(|| {
                    KvError::InvalidDimensions("elem_start overflowed usize".to_owned())
                })?;
                let elem_end = end_pos.checked_mul(slab.kv_dim).ok_or_else(|| {
                    KvError::InvalidDimensions("elem_end overflowed usize".to_owned())
                })?;
                let (k_plane, v_plane) = view
                    .layer_kv_mut(layer)
                    .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
                k_plane[elem_start..elem_end].copy_from_slice(k_batch);
                v_plane[elem_start..elem_end].copy_from_slice(v_batch);
                Ok((&k_plane[..elem_end], &v_plane[..elem_end]))
            }
            WriterBacking::Bytes { kv_bytes, slab } => {
                write_kv_f32_planes(kv_bytes, *slab, layer, start_pos, k_batch, v_batch)
            }
        }
    }

    /// Packs and writes contiguous `[seq_len, kv_dim]` `f32` K and V batch slices into `layer`
    /// starting at `start_pos` using the slab's [`KvDtype`] (`F32`, `F16`, or `Q8_0`), returning
    /// the updated exclusive end token position (`start_pos + seq_len`).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `layer >= num_layers`, `k_batch.len() != v_batch.len()`,
    /// lengths are not multiples of `kv_dim`, `start_pos + seq_len > seq_capacity`, or `Q8_0`
    /// packing bounds fail.
    pub fn append_quantized_kv_batch(
        &mut self,
        layer: usize,
        start_pos: usize,
        k_batch: &[f32],
        v_batch: &[f32],
    ) -> Result<usize, KvError> {
        let slab = self.slab();
        let (_seq_len, end_pos) =
            slab.validate_batch_write(layer, start_pos, k_batch.len(), v_batch.len())?;
        match &mut self.backing {
            WriterBacking::F32View(view) => {
                let (k_plane, v_plane) = view
                    .layer_kv_mut(layer)
                    .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
                pack_kv_batch_plane(
                    KvDtype::F32,
                    slab.kv_dim,
                    start_pos,
                    k_batch,
                    k_plane.as_mut_bytes(),
                )?;
                pack_kv_batch_plane(
                    KvDtype::F32,
                    slab.kv_dim,
                    start_pos,
                    v_batch,
                    v_plane.as_mut_bytes(),
                )?;
            }
            WriterBacking::Bytes { kv_bytes, slab } => {
                let (k_range, v_range) = slab.layer_plane_bounds(layer)?;
                if v_range.end > kv_bytes.len() {
                    return Err(KvError::InvalidDimensions(
                        "KV layer plane range exceeds slab byte length".to_owned(),
                    ));
                }
                let (prefix, tail) = kv_bytes.split_at_mut(v_range.start);
                let k_plane = &mut prefix[k_range];
                let v_plane = &mut tail[..v_range.len()];
                pack_kv_batch_plane(slab.kv_dtype, slab.kv_dim, start_pos, k_batch, k_plane)?;
                pack_kv_batch_plane(slab.kv_dtype, slab.kv_dim, start_pos, v_batch, v_plane)?;
            }
        }
        Ok(end_pos)
    }

    /// Returns raw aligned K and V plane byte slices for `layer`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `layer >= num_layers` or plane bounds fail.
    pub fn layer_kv_bytes(&self, layer: usize) -> Result<(&[u8], &[u8]), KvError> {
        match &self.backing {
            WriterBacking::F32View(view) => {
                let (k_plane, v_plane) = view
                    .layer_kv(layer)
                    .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
                Ok((k_plane.as_bytes(), v_plane.as_bytes()))
            }
            WriterBacking::Bytes { kv_bytes, slab } => read_kv_bytes_planes(kv_bytes, *slab, layer),
        }
    }

    /// Unpacks `[start_pos..end_pos)` rows from `layer` into caller-provided `k_out` and `v_out`
    /// `f32` buffers without heap allocation, returning `(k_slice, v_slice)`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if bounds, `layer`, or output buffer capacities fail.
    pub fn read_kv_range_f32<'o>(
        &self,
        layer: usize,
        start_pos: usize,
        end_pos: usize,
        k_out: &'o mut [f32],
        v_out: &'o mut [f32],
    ) -> Result<(&'o [f32], &'o [f32]), KvError> {
        let slab = self.slab();
        let (_seq_len, _elem_len) = slab.validate_pos_range(start_pos, end_pos)?;
        let (k_plane, v_plane) = self.layer_kv_bytes(layer)?;
        unpack_kv_planes_range_f32(slab, k_plane, v_plane, start_pos..end_pos, k_out, v_out)
    }

    /// Unpacks the active window up to `end_pos` (respecting `sliding_window` when configured)
    /// into caller-provided `k_out` and `v_out` buffers without heap allocation.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if bounds, `layer`, or output buffer capacities fail.
    pub fn read_valid_kv_range_f32<'o>(
        &self,
        layer: usize,
        end_pos: usize,
        k_out: &'o mut [f32],
        v_out: &'o mut [f32],
    ) -> Result<(&'o [f32], &'o [f32]), KvError> {
        let start_pos = self.slab().valid_start_pos(end_pos);
        self.read_kv_range_f32(layer, start_pos, end_pos, k_out, v_out)
    }

    /// Unpacks the currently committed valid range (`valid_kv_range(committed_tokens())`) into
    /// caller-provided `k_out` and `v_out` buffers without heap allocation.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if metadata, bounds, or output buffers fail.
    pub fn read_committed_kv_range_f32<'o>(
        &self,
        layer: usize,
        k_out: &'o mut [f32],
        v_out: &'o mut [f32],
    ) -> Result<(&'o [f32], &'o [f32]), KvError> {
        let committed = self.committed_tokens()?;
        self.read_valid_kv_range_f32(layer, committed, k_out, v_out)
    }

    /// Zero-allocation unpack of `[start_pos..end_pos)` into 128-byte aligned buffers allocated
    /// from `scope`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if bounds fail or scratch capacity is exceeded.
    pub fn read_kv_range_in_scratch<'s>(
        &self,
        layer: usize,
        start_pos: usize,
        end_pos: usize,
        scope: &'s ScratchScope<'_>,
    ) -> Result<(&'s [f32], &'s [f32]), KvError> {
        let slab = self.slab();
        let (k_plane, v_plane) = self.layer_kv_bytes(layer)?;
        unpack_kv_planes_in_scratch(slab, k_plane, v_plane, start_pos, end_pos, scope)
    }

    /// Zero-allocation unpack of the currently committed valid KV window into 128-byte aligned
    /// buffers allocated from `scope`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if metadata or bounds fail, or scratch capacity is exceeded.
    pub fn read_committed_kv_range_in_scratch<'s>(
        &self,
        layer: usize,
        scope: &'s ScratchScope<'_>,
    ) -> Result<(&'s [f32], &'s [f32]), KvError> {
        let committed = self.committed_tokens()?;
        let start_pos = self.slab().valid_start_pos(committed);
        self.read_kv_range_in_scratch(layer, start_pos, committed, scope)
    }

    /// Returns contiguous `0..end_pos` immutable `(k_prefix, v_prefix)` slices for `layer` (`F32` only).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `kv_dtype != F32`, `layer >= num_layers`, or
    /// `end_pos > seq_capacity`.
    pub fn layer_kv_prefix(
        &self,
        layer: usize,
        end_pos: usize,
    ) -> Result<(&[f32], &[f32]), KvError> {
        let slab = self.slab();
        match &self.backing {
            WriterBacking::F32View(view) => {
                let (_seq_len, elem_end) = slab.validate_pos_range(0, end_pos)?;
                let (k_plane, v_plane) = view
                    .layer_kv(layer)
                    .map_err(|err| KvError::InvalidDimensions(err.to_string()))?;
                Ok((&k_plane[..elem_end], &v_plane[..elem_end]))
            }
            WriterBacking::Bytes { kv_bytes, slab } => {
                read_kv_f32_range(kv_bytes, *slab, layer, 0, end_pos)
            }
        }
    }

    /// Returns immutable `(k_valid, v_valid)` slices for `layer` up to `end_pos` (respecting
    /// `sliding_window` when configured, `F32` only).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `kv_dtype != F32`, `layer >= num_layers`, or
    /// `end_pos > seq_capacity`.
    pub fn valid_kv_range(
        &self,
        layer: usize,
        end_pos: usize,
    ) -> Result<(&[f32], &[f32]), KvError> {
        let slab = self.slab();
        match &self.backing {
            WriterBacking::F32View(view) => view
                .valid_kv_range(layer, end_pos)
                .map_err(|err| KvError::InvalidDimensions(err.to_string())),
            WriterBacking::Bytes { kv_bytes, slab: s } => {
                read_kv_f32_range(kv_bytes, *s, layer, slab.valid_start_pos(end_pos), end_pos)
            }
        }
    }

    /// Reads the `committed_token_count` for `block_idx` with `Acquire` ordering.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `block_idx >= block_count()`.
    pub fn block_committed_tokens(&self, block_idx: usize) -> Result<u32, KvError> {
        read_block_committed_tokens_from_slice(self.meta_bytes, block_idx, self.logical_block_count)
    }

    /// Computes the total contiguously committed token watermark across all 16-token blocks.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if any block exceeds 16 tokens or blocks are non-contiguous.
    pub fn committed_tokens(&self) -> Result<usize, KvError> {
        committed_watermark_from_meta(self.meta_bytes, self.logical_block_count)
    }

    /// Enforces monotonic contiguous progression (`start_pos == committed_tokens()`) and advances
    /// [`KvBlockMeta::committed_token_count`] with `Release` ordering across all 16-token blocks
    /// spanned by `[start_pos .. start_pos + seq_len)`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `start_pos != committed_tokens()`, `start_pos + seq_len`
    /// overflows or exceeds `seq_capacity`, or existing block metadata is corrupted/non-contiguous.
    pub fn commit_token_span(&mut self, start_pos: usize, seq_len: usize) -> Result<(), KvError> {
        commit_token_span_in_meta(
            self.meta_bytes,
            self.logical_block_count,
            self.seq_capacity(),
            start_pos,
            seq_len,
        )
    }

    /// Commits a batched prefill range `[start_pos .. start_pos + seq_len)` across spanned 16-token
    /// blocks via [`Self::commit_token_span`].
    ///
    /// # Errors
    /// Returns [`KvError`] if [`Self::commit_token_span`] fails.
    pub fn commit_prefill_range(
        &mut self,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<(), KvError> {
        self.commit_token_span(start_pos, seq_len)
    }
}

/// Zero-copy read-only view over an initialized arena's KV cache slab and atomic [`KvBlockMeta`] table.
#[derive(Debug)]
pub struct ArenaKvReader<'a> {
    kv_bytes: &'a [u8],
    meta_bytes: &'a [u8],
    slab: KvSlabGeometry,
    logical_block_count: usize,
}

impl<'a> ArenaKvReader<'a> {
    /// Wraps an initialized arena byte slice in a bounds-checked [`ArenaKvReader`].
    ///
    /// Supports `F32`, `F16`, and `Q8_0` KV slabs aligned to [`VECTOR_PLANE_ALIGNMENT`].
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if header, KV slab, or metadata checks fail, or
    /// [`KvError::IpcError`] if metadata alignment is invalid for `AtomicU32` loads.
    pub fn from_arena_slice(arena_bytes: &'a [u8]) -> Result<Self, KvError> {
        let validated = validate_arena_kv_slab(arena_bytes)?;
        let (meta_offset, meta_end, logical_block_count) = validate_arena_meta_bounds(
            arena_bytes,
            validated.slab.geometry.seq_capacity,
            validated.kv_offset,
        )?;
        let meta_bytes = &arena_bytes[meta_offset..meta_end];
        validate_meta_slice_for_atomic(meta_bytes, logical_block_count)?;
        let kv_bytes = &arena_bytes[validated.kv_offset..validated.kv_end];

        Ok(Self {
            kv_bytes,
            meta_bytes,
            slab: validated.slab,
            logical_block_count,
        })
    }

    /// Wraps a [`SharedMemoryArena`] in a read-only [`ArenaKvReader`].
    ///
    /// # Errors
    /// Returns [`KvError`] if header, KV slab, or metadata validation fails.
    pub fn from_arena(arena: &'a SharedMemoryArena) -> Result<Self, KvError> {
        Self::from_arena_slice(arena.as_slice())
    }

    /// Returns the validated [`KvGeometry`].
    #[must_use]
    pub const fn geometry(&self) -> KvGeometry {
        self.slab.geometry
    }

    /// Returns the number of transformer KV layers.
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.slab.geometry.num_layers
    }

    /// Returns the maximum sequence token capacity (`seq_capacity`).
    #[must_use]
    pub const fn seq_capacity(&self) -> usize {
        self.slab.geometry.seq_capacity
    }

    /// Returns the per-position vector width (`n_heads_kv * head_dim`).
    #[must_use]
    pub const fn kv_dim(&self) -> usize {
        self.slab.kv_dim
    }

    /// Returns the element encoding ([`KvDtype`]) of the underlying KV slab.
    #[must_use]
    pub const fn kv_dtype(&self) -> KvDtype {
        self.slab.kv_dtype
    }

    /// Returns the encoded byte stride per token row (`KvDtype::row_bytes(kv_dim)`).
    #[must_use]
    pub const fn kv_row_bytes(&self) -> usize {
        self.slab.kv_row_bytes
    }

    /// Returns the aligned byte stride between consecutive K/V planes.
    #[must_use]
    pub const fn kv_plane_stride_bytes(&self) -> usize {
        self.slab.kv_plane_stride_bytes
    }

    /// Returns the semantic element length of a single K or V plane (`seq_capacity * kv_dim`).
    #[must_use]
    pub const fn plane_len(&self) -> usize {
        self.slab.plane_len()
    }

    /// Returns the number of 16-token blocks backing `seq_capacity`.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.logical_block_count
    }

    /// Returns raw aligned `(k_plane_bytes, v_plane_bytes)` slices for `layer`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `layer >= num_layers()` or plane bounds fail.
    pub fn layer_kv_bytes(&self, layer: usize) -> Result<(&'a [u8], &'a [u8]), KvError> {
        read_kv_bytes_planes(self.kv_bytes, self.slab, layer)
    }

    /// Unpacks `[start_pos..end_pos)` rows from `layer` into caller-provided `k_out` and `v_out`
    /// `f32` buffers without heap allocation, returning `(k_slice, v_slice)`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if bounds, `layer`, or output buffer capacities fail.
    pub fn read_kv_range_f32<'o>(
        &self,
        layer: usize,
        start_pos: usize,
        end_pos: usize,
        k_out: &'o mut [f32],
        v_out: &'o mut [f32],
    ) -> Result<(&'o [f32], &'o [f32]), KvError> {
        let (_seq_len, _elem_len) = self.slab.validate_pos_range(start_pos, end_pos)?;
        let (k_plane, v_plane) = self.layer_kv_bytes(layer)?;
        unpack_kv_planes_range_f32(
            self.slab,
            k_plane,
            v_plane,
            start_pos..end_pos,
            k_out,
            v_out,
        )
    }

    /// Unpacks the active window up to `end_pos` (respecting `sliding_window` when configured)
    /// into caller-provided `k_out` and `v_out` buffers without heap allocation.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if bounds, `layer`, or output buffer capacities fail.
    pub fn read_valid_kv_range_f32<'o>(
        &self,
        layer: usize,
        end_pos: usize,
        k_out: &'o mut [f32],
        v_out: &'o mut [f32],
    ) -> Result<(&'o [f32], &'o [f32]), KvError> {
        let start_pos = self.slab.valid_start_pos(end_pos);
        self.read_kv_range_f32(layer, start_pos, end_pos, k_out, v_out)
    }

    /// Unpacks the currently committed valid range (`valid_kv_range(committed_tokens())`) into
    /// caller-provided `k_out` and `v_out` buffers without heap allocation.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if metadata, bounds, or output buffers fail.
    pub fn read_committed_kv_range_f32<'o>(
        &self,
        layer: usize,
        k_out: &'o mut [f32],
        v_out: &'o mut [f32],
    ) -> Result<(&'o [f32], &'o [f32]), KvError> {
        let committed = self.committed_tokens()?;
        self.read_valid_kv_range_f32(layer, committed, k_out, v_out)
    }

    /// Zero-allocation unpack of `[start_pos..end_pos)` into 128-byte aligned buffers allocated
    /// from `scope`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if bounds fail or scratch capacity is exceeded.
    pub fn read_kv_range_in_scratch<'s>(
        &self,
        layer: usize,
        start_pos: usize,
        end_pos: usize,
        scope: &'s ScratchScope<'_>,
    ) -> Result<(&'s [f32], &'s [f32]), KvError> {
        let (k_plane, v_plane) = self.layer_kv_bytes(layer)?;
        unpack_kv_planes_in_scratch(self.slab, k_plane, v_plane, start_pos, end_pos, scope)
    }

    /// Zero-allocation unpack of the currently committed valid KV window into 128-byte aligned
    /// buffers allocated from `scope`.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if metadata or bounds fail, or scratch capacity is exceeded.
    pub fn read_committed_kv_range_in_scratch<'s>(
        &self,
        layer: usize,
        scope: &'s ScratchScope<'_>,
    ) -> Result<(&'s [f32], &'s [f32]), KvError> {
        let committed = self.committed_tokens()?;
        let start_pos = self.slab.valid_start_pos(committed);
        self.read_kv_range_in_scratch(layer, start_pos, committed, scope)
    }

    /// Returns immutable `(k_plane, v_plane)` slices covering full `[0..seq_capacity)` for `layer` (`F32` only).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `kv_dtype != F32` or `layer >= num_layers()`.
    pub fn layer_kv(&self, layer: usize) -> Result<(&'a [f32], &'a [f32]), KvError> {
        read_kv_f32_planes(self.kv_bytes, self.slab, layer)
    }

    /// Returns immutable `(k_slice, v_slice)` covering `[start_pos..end_pos)` for `layer` (`F32` only).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `kv_dtype != F32`, `layer >= num_layers()`,
    /// `start_pos > end_pos`, or `end_pos > seq_capacity()`.
    pub fn layer_kv_range(
        &self,
        layer: usize,
        start_pos: usize,
        end_pos: usize,
    ) -> Result<(&'a [f32], &'a [f32]), KvError> {
        read_kv_f32_range(self.kv_bytes, self.slab, layer, start_pos, end_pos)
    }

    /// Returns immutable `(k_prefix, v_prefix)` slices covering `0..end_pos` for `layer` (`F32` only).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `kv_dtype != F32`, `layer >= num_layers()`, or
    /// `end_pos > seq_capacity()`.
    pub fn layer_kv_prefix(
        &self,
        layer: usize,
        end_pos: usize,
    ) -> Result<(&'a [f32], &'a [f32]), KvError> {
        self.layer_kv_range(layer, 0, end_pos)
    }

    /// Returns immutable `(k_valid, v_valid)` slices up to `end_pos` (respecting `sliding_window`
    /// when configured, `F32` only).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `kv_dtype != F32`, `layer >= num_layers()`, or
    /// `end_pos > seq_capacity()`.
    pub fn valid_kv_range(
        &self,
        layer: usize,
        end_pos: usize,
    ) -> Result<(&'a [f32], &'a [f32]), KvError> {
        let start_pos = self.slab.valid_start_pos(end_pos);
        self.layer_kv_range(layer, start_pos, end_pos)
    }

    /// Reads the current committed watermark and returns `valid_kv_range(layer, committed_tokens)` (`F32` only).
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if metadata is corrupt/non-contiguous, `kv_dtype != F32`,
    /// or `layer >= num_layers()`.
    pub fn committed_kv_range(&self, layer: usize) -> Result<(&'a [f32], &'a [f32]), KvError> {
        let committed = self.committed_tokens()?;
        self.valid_kv_range(layer, committed)
    }

    /// Reads the `committed_token_count` for `block_idx` with `Acquire` ordering.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if `block_idx >= block_count()`.
    pub fn block_committed_tokens(&self, block_idx: usize) -> Result<u32, KvError> {
        read_block_committed_tokens_from_slice(self.meta_bytes, block_idx, self.logical_block_count)
    }

    /// Computes the total contiguously committed token watermark across all 16-token blocks.
    ///
    /// # Errors
    /// Returns [`KvError::InvalidDimensions`] if any block exceeds 16 tokens or blocks are non-contiguous.
    pub fn committed_tokens(&self) -> Result<usize, KvError> {
        committed_watermark_from_meta(self.meta_bytes, self.logical_block_count)
    }
}

/// Constructs a bounds-checked [`KvCacheView`] over an initialized arena byte buffer.
///
/// Exposed alongside [`ArenaHandle`] so layer-major `[layer][K|V][pos][kv_dim]` invariants can
/// also be verified directly under Miri without OS `memfd_create` / `mmap` syscalls.
///
/// # Errors
/// Returns [`KvError::InvalidDimensions`] if header validation fails, geometry or dtype checks
/// fail, planes have alignment padding, or the KV slab range violates header region bounds or `f32` alignment.
pub fn kv_cache_view_from_arena_slice_mut(
    arena_bytes: &mut [u8],
) -> Result<KvCacheView<'_>, KvError> {
    let validated = validate_arena_kv_slab(arena_bytes)?;
    if validated.slab.kv_dtype != KvDtype::F32 {
        return Err(KvError::InvalidDimensions(format!(
            "KvCacheView requires F32 KV cache (kv_dtype=0), got {:?}",
            validated.slab.kv_dtype
        )));
    }
    if validated.slab.kv_plane_stride_bytes != validated.slab.raw_plane_bytes {
        return Err(KvError::InvalidDimensions(format!(
            "KvCacheView requires unpadded contiguous F32 planes (raw_plane_bytes={}, kv_plane_stride_bytes={})",
            validated.slab.raw_plane_bytes, validated.slab.kv_plane_stride_bytes
        )));
    }
    let kv_bytes = &mut arena_bytes[validated.kv_offset..validated.kv_end];
    let kv_f32s = <[f32]>::mut_from_bytes(kv_bytes).map_err(|err| {
        KvError::InvalidDimensions(format!("KV slab byte slice conversion failed: {err:?}"))
    })?;
    KvCacheView::new(kv_f32s, validated.slab.geometry)
        .map_err(|err| KvError::InvalidDimensions(err.to_string()))
}

#[cfg(test)]
mod tests {
    use jero_core::ipc::protocol::{KV_DTYPE_F16, KV_DTYPE_Q8_0};

    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
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
    #[cfg_attr(miri, ignore)]
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

    #[test]
    fn test_compute_hybrid_kv_block_count() {
        assert_eq!(compute_hybrid_kv_block_count(0, &[]), 0);

        // Gemma 3 / 4: 36 layers in 5:1 local(1024) : global(n_ctx) pattern -> 30 local + 6 global.
        let gemma3_policies: Vec<LayerKvPolicy> = (0..36)
            .map(|l| {
                if (l + 1) % 6 == 0 {
                    LayerKvPolicy::GlobalFullContext
                } else {
                    LayerKvPolicy::SlidingWindow {
                        window_tokens: 1024,
                    }
                }
            })
            .collect();
        // 30 * (1024 / 16) + 6 * (32768 / 16) = 30 * 64 + 6 * 2048 = 1920 + 12288 = 14208.
        assert_eq!(
            compute_hybrid_kv_block_count(32_768, &gemma3_policies),
            14_208
        );

        // Gemma 2: 26 layers in 1:1 alternating local(4096) : global(8192).
        let gemma2_policies: Vec<LayerKvPolicy> = (0..26)
            .map(|l| {
                if l % 2 == 0 {
                    LayerKvPolicy::SlidingWindow {
                        window_tokens: 4096,
                    }
                } else {
                    LayerKvPolicy::GlobalFullContext
                }
            })
            .collect();
        // 13 * 256 + 13 * 512 = 9984.
        assert_eq!(
            compute_hybrid_kv_block_count(8_192, &gemma2_policies),
            9_984
        );

        // Arena layout with hybrid policies allocates strictly fewer bytes than homogeneous layout.
        let full_cfg = ArenaAllocationConfig::new("gemma3", 32_768, 256, 36).with_head_count_kv(4);
        let hybrid_cfg = ArenaAllocationConfig::new("gemma3", 32_768, 256, 36)
            .with_head_count_kv(4)
            .with_layer_policies(gemma3_policies);
        let full_layout = compute_arena_layout(&full_cfg).expect("full layout");
        let hybrid_layout = compute_arena_layout(&hybrid_cfg).expect("hybrid layout");
        assert!(
            hybrid_layout.total_bytes * 4 < full_layout.total_bytes,
            "hybrid 5:1 arena ({}) should be >4x smaller than homogeneous full-context ({})",
            hybrid_layout.total_bytes,
            full_layout.total_bytes
        );
    }

    #[test]
    fn test_holder_mla_latent_stride() {
        let cfg = ArenaAllocationConfig::new("deepseek2", 64, 192, 27)
            .with_head_count_kv(16)
            .with_kv_latent_dim(512, 64);
        assert_eq!(cfg.kv_latent_dim, Some(576));
        let layout = compute_arena_layout(&cfg).expect("mla layout");
        let zero_cfg = ArenaAllocationConfig::new("deepseek2", 64, 0, 1).with_kv_latent_dim(0, 0);
        assert!(matches!(
            compute_arena_layout(&zero_cfg),
            Err(KvError::InvalidDimensions(_))
        ));
        assert!(layout.total_bytes > 0);
    }

    #[test]
    fn test_mla_compressed_block_bytes() {
        // Single layer, single 16-token block: 16 * (512 + 64) * 4 = 36_864 bytes of KV block payload
        let cfg = ArenaAllocationConfig::new("deepseek2", 16, 0, 1)
            .with_vocab_size(0)
            .with_kv_latent_dim(512, 64);
        let layout = compute_arena_layout(&cfg).expect("single-block mla layout");
        let kv_span = layout.output_slabs_offset - layout.kv_blocks_offset;
        assert_eq!(kv_span, 36_864);
    }

    #[test]
    fn test_deepseek2_mla_latent_kv_layout() {
        // DeepSeek-V2-Lite: 27 layers, n_ctx = 4096, uncompressed 16*192 vs latent 512+64=576
        let uncompressed_cfg =
            ArenaAllocationConfig::new("deepseek2", 4096, 192, 27).with_head_count_kv(16);
        let mla_cfg = ArenaAllocationConfig::new("deepseek2", 4096, 192, 27)
            .with_head_count_kv(16)
            .with_kv_latent_dim(512, 64);
        let full_layout = compute_arena_layout(&uncompressed_cfg).expect("full layout");
        let mla_layout = compute_arena_layout(&mla_cfg).expect("mla layout");
        assert!(
            mla_layout.total_bytes * 5 < full_layout.total_bytes,
            "MLA latent arena ({}) should be >80% smaller than uncompressed GQA ({})",
            mla_layout.total_bytes,
            full_layout.total_bytes
        );
    }

    #[test]
    fn test_o1_recurrent_state_arena_sizing() {
        let small_ctx_cfg =
            ArenaAllocationConfig::new_recurrent("mamba2", 512, 32_000, 64, 1024, 4096);
        let large_ctx_cfg =
            ArenaAllocationConfig::new_recurrent("mamba2", 32_768, 32_000, 64, 1024, 4096);

        let small_layout = compute_arena_layout(&small_ctx_cfg).expect("512-ctx recurrent layout");
        let large_layout = compute_arena_layout(&large_ctx_cfg).expect("32k-ctx recurrent layout");

        assert_eq!(small_layout.block_count, 0);
        assert_eq!(large_layout.block_count, 0);
        assert_eq!(
            small_layout.total_bytes, large_layout.total_bytes,
            "pure recurrent arena sizing must remain O(1) as n_ctx grows from 512 to 32768"
        );
        assert_eq!(small_layout.recurrent_bank_stride_bytes, 64 * (1024 + 4096));
        assert_eq!(small_layout.recurrent_total_bytes, 64 + 2 * 64 * 5120);

        let hybrid_cfg = ArenaAllocationConfig::new("jamba", 1024, 128, 4)
            .with_head_count_kv(4)
            .with_vocab_size(32_000)
            .with_recurrent_state(28, 512, 2048);
        let hybrid_layout = compute_arena_layout(&hybrid_cfg).expect("hybrid jamba layout");
        assert_eq!(hybrid_layout.block_count, 64);
        assert!(hybrid_layout.recurrent_state_offset >= hybrid_layout.output_slabs_offset);
        assert_eq!(hybrid_layout.recurrent_bank_stride_bytes, 28 * (512 + 2048));
        assert_eq!(hybrid_layout.recurrent_total_bytes, 64 + 2 * 28 * 2560);
    }

    #[test]
    fn test_ping_pong_bank_flip_and_rollback() {
        #[repr(C, align(64))]
        struct AlignedRegion([u8; 4096]);

        let mut region = AlignedRegion([0_u8; 4096]);
        let init = RecurrentStateSlotHeader::new(2, 64, 128, 384);
        assert_eq!(init.bank_stride_bytes, 384);
        assert_eq!(init.total_region_bytes(), 832);
        region.0[..size_of::<RecurrentStateSlotHeader>()].copy_from_slice(init.as_bytes());

        {
            let (hdr, active, inactive) =
                recurrent_banks_from_region_mut(&mut region.0).expect("initial bank split");
            assert_eq!(hdr.committed_step, 0);
            assert_eq!(hdr.active_bank, 0);
            assert!(active.iter().all(|&b| b == 0));
            inactive.fill(0xA1);
        }

        let step_one_hdr =
            commit_recurrent_step_in_region(&mut region.0, 1).expect("commit step 1");
        assert_eq!(step_one_hdr.committed_step, 1);
        assert_eq!(step_one_hdr.active_bank, 1);

        // Simulate a worker crash partway through writing the staging bank before atomic commit.
        {
            let (hdr, active, inactive) =
                recurrent_banks_from_region_mut(&mut region.0).expect("step 1 bank split");
            assert_eq!(hdr.committed_step, 1);
            assert_eq!(hdr.active_bank, 1);
            assert!(active.iter().all(|&b| b == 0xA1));
            inactive[..128].fill(0xEE);
        }

        // Rollback verification: committed_step, active_bank, and bank 1 contents remain intact.
        let reread = read_recurrent_slot_header_from_region(&region.0).expect("reread header");
        assert_eq!(reread.committed_step, 1);
        assert_eq!(reread.active_bank, 1);
        {
            let (_, active, inactive) =
                recurrent_banks_from_region_mut(&mut region.0).expect("post-crash bank split");
            assert!(
                active.iter().all(|&b| b == 0xA1),
                "active committed bank must never tear on incomplete staging write"
            );
            inactive.fill(0xB2);
        }

        let step_two_hdr =
            commit_recurrent_step_in_region(&mut region.0, 2).expect("commit step 2");
        assert_eq!(step_two_hdr.committed_step, 2);
        assert_eq!(step_two_hdr.active_bank, 0);
        let (hdr_after, active_after, _) =
            recurrent_banks_from_region_mut(&mut region.0).expect("step 2 bank split");
        assert_eq!(hdr_after.committed_step, 2);
        assert_eq!(hdr_after.active_bank, 0);
        assert!(active_after.iter().all(|&b| b == 0xB2));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_ping_pong_recurrent_checkpoint() {
        let holder = KvHolder::new();
        let cfg = ArenaAllocationConfig::new_recurrent("rwkv7", 256, 1024, 4, 128, 256);
        let res = holder
            .allocate(&cfg)
            .expect("pure recurrent allocation should succeed");
        assert_eq!(res.block_count, 0);

        let initial_hdr = holder
            .read_recurrent_slot_header(res.arena_id)
            .expect("recurrent header should be initialized");
        assert_eq!(initial_hdr.committed_step, 0);
        assert_eq!(initial_hdr.active_bank, 0);
        assert_eq!(initial_hdr.num_recurrent_layers, 4);
        assert_eq!(initial_hdr.conv_state_bytes_per_layer, 128);
        assert_eq!(initial_hdr.ssm_state_bytes_per_layer, 256);

        let fd = holder.get_arena_fd(res.arena_id).expect("arena fd");
        let len = usize::try_from(res.total_bytes).expect("arena len fits usize");
        let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "test_rec_rw", false)
            .expect("rw mapping from existing fd");

        {
            let (hdr, active, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("arena bank split");
            assert_eq!(hdr.active_bank, 0);
            assert!(active.iter().all(|&b| b == 0));
            inactive.fill(0x3C);
        }

        let committed =
            commit_recurrent_step_in_arena(rw.as_mut_slice(), 7).expect("commit step 7");
        assert_eq!(committed.committed_step, 7);
        assert_eq!(committed.active_bank, 1);

        let (snap_hdr, snap_bank) = holder
            .read_active_recurrent_bank(res.arena_id)
            .expect("read committed active bank");
        assert_eq!(snap_hdr.committed_step, 7);
        assert_eq!(snap_hdr.active_bank, 1);
        assert_eq!(
            snap_bank.len(),
            usize::try_from(snap_hdr.bank_stride_bytes).unwrap()
        );
        assert!(snap_bank.iter().all(|&b| b == 0x3C));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_atomic_ping_pong_recurrent_recovery() {
        let holder = KvHolder::new();
        let cfg = ArenaAllocationConfig::new_recurrent("mamba2", 512, 2048, 2, 64, 192);
        let res = holder
            .allocate(&cfg)
            .expect("recurrent arena allocation should succeed");
        let fd = holder.get_arena_fd(res.arena_id).expect("arena fd");
        let len = usize::try_from(res.total_bytes).expect("arena len fits usize");
        let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "test_rec_recovery", false)
            .expect("rw mapping");

        {
            let (_, _, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("bank split step 1");
            inactive.fill(0x11);
        }
        let first_commit =
            commit_recurrent_step_in_arena(rw.as_mut_slice(), 1).expect("commit step 1");
        assert_eq!(first_commit.committed_step, 1);
        assert_eq!(first_commit.active_bank, 1);

        // Write a partial torn state to inactive bank 0 and simulate crash before commit.
        {
            let (_, _, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("bank split torn");
            inactive[..64].fill(0xFF);
        }

        let (recovered_hdr, recovered_bank) = holder
            .read_active_recurrent_bank(res.arena_id)
            .expect("recovery snapshot");
        assert_eq!(recovered_hdr.committed_step, 1);
        assert_eq!(recovered_hdr.active_bank, 1);
        assert!(recovered_bank.iter().all(|&b| b == 0x11));

        // Resume cleanly and commit step 2.
        {
            let (_, _, inactive) =
                recurrent_banks_in_arena_mut(rw.as_mut_slice()).expect("bank split step 2");
            inactive.fill(0x22);
        }
        let second_commit =
            commit_recurrent_step_in_arena(rw.as_mut_slice(), 2).expect("commit step 2");
        assert_eq!(second_commit.committed_step, 2);
        assert_eq!(second_commit.active_bank, 0);
        let (final_hdr, final_bank) = holder
            .read_active_recurrent_bank(res.arena_id)
            .expect("final snapshot");
        assert_eq!(final_hdr.committed_step, 2);
        assert_eq!(final_hdr.active_bank, 0);
        assert!(final_bank.iter().all(|&b| b == 0x22));
    }

    #[test]
    fn test_from_request_hydrates_real_geometry_and_fails_closed() {
        let req = AllocateArenaRequest {
            model_id: "qwen2.5-7b-instruct".to_owned(),
            arena_bytes: 16 * 1024 * 1024,
            n_ctx: 2048,
            max_blocks: 128,
            num_layers: 28,
            head_count_kv: 8,
            head_dim: 128,
            vocab_size: 128_256,
            kv_dtype: KV_DTYPE_F16,
        };
        let cfg =
            ArenaAllocationConfig::from_request(&req).expect("valid request hydrates cleanly");
        assert_eq!(cfg.model_id, "qwen2.5-7b-instruct");
        assert_eq!(cfg.n_ctx, 2048);
        assert_eq!(cfg.max_blocks, Some(128));
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.head_count_kv, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 128_256);
        assert_eq!(cfg.kv_dtype, KV_DTYPE_F16);
        assert_eq!(cfg.explicit_bytes, Some(16 * 1024 * 1024));

        // Fail closed on every zeroed geometry/capacity dimension and invalid kv_dtype.
        let invalid_cases = [
            AllocateArenaRequest {
                model_id: "   ".to_owned(),
                ..req.clone()
            },
            AllocateArenaRequest {
                arena_bytes: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                n_ctx: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                max_blocks: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                num_layers: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                head_count_kv: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                head_dim: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                vocab_size: 0,
                ..req.clone()
            },
            AllocateArenaRequest {
                kv_dtype: 99,
                ..req
            },
        ];

        for invalid in &invalid_cases {
            let err = ArenaAllocationConfig::from_request(invalid)
                .expect_err("zero/invalid dimension must fail closed");
            assert!(
                matches!(err, KvError::InvalidDimensions(_)),
                "expected InvalidDimensions, got {err:?}"
            );
        }
    }

    #[test]
    fn test_compute_arena_layout_rejects_zero_kv_dims_and_bad_dtype() {
        let zero_heads = ArenaAllocationConfig::new("m", 64, 64, 2).with_head_count_kv(0);
        assert!(matches!(
            compute_arena_layout(&zero_heads),
            Err(KvError::InvalidDimensions(_))
        ));

        let zero_vocab = ArenaAllocationConfig::new("m", 64, 64, 2).with_vocab_size(0);
        assert!(matches!(
            compute_arena_layout(&zero_vocab),
            Err(KvError::InvalidDimensions(_))
        ));

        let bad_dtype = ArenaAllocationConfig::new("m", 64, 64, 2).with_kv_dtype(3);
        assert!(matches!(
            compute_arena_layout(&bad_dtype),
            Err(KvError::InvalidDimensions(_))
        ));

        // Verify F16 layout allocates 2 bytes per KV scalar instead of 4.
        let f32_cfg = ArenaAllocationConfig::new("m", 32, 64, 4)
            .with_head_count_kv(4)
            .with_vocab_size(128)
            .with_kv_dtype(KV_DTYPE_F32);
        let f16_cfg = ArenaAllocationConfig::new("m", 32, 64, 4)
            .with_head_count_kv(4)
            .with_vocab_size(128)
            .with_kv_dtype(KV_DTYPE_F16);
        let f32_layout = compute_arena_layout(&f32_cfg).expect("f32 layout");
        let f16_layout = compute_arena_layout(&f16_cfg).expect("f16 layout");
        assert_eq!(f32_layout.seq_capacity, 32);
        assert_eq!(f16_layout.seq_capacity, 32);
        let f32_span = f32_layout.output_slabs_offset - f32_layout.kv_blocks_offset;
        let f16_span = f16_layout.output_slabs_offset - f16_layout.kv_blocks_offset;
        assert_eq!(f32_span, 4 * 2 * 32 * 4 * 64 * 4);
        assert_eq!(f16_span, 4 * 2 * 32 * 4 * 64 * 2);

        // Verify Q8_0 layout uses 34 bytes per 32-element group.
        let q8_cfg = ArenaAllocationConfig::new("m", 32, 64, 4)
            .with_head_count_kv(4)
            .with_vocab_size(128)
            .with_kv_dtype(KV_DTYPE_Q8_0);
        let q8_layout = compute_arena_layout(&q8_cfg).expect("q8_0 layout");
        let kv_dim = 4_u64 * 64;
        let expected_q8_raw = 4 * 2 * 32 * (kv_dim / 32) * 34;
        assert_eq!(
            q8_layout.output_slabs_offset - q8_layout.kv_blocks_offset,
            align_up_u64(expected_q8_raw, VECTOR_PLANE_ALIGNMENT as u64)
        );
    }

    #[test]
    fn test_layer_major_kv_cache_view_writes_and_reads_without_seq_cap_drift() {
        #[repr(C, align(64))]
        #[derive(Clone, Copy, FromBytes, IntoBytes, zerocopy::Immutable)]
        struct AlignedLine([u8; 64]);

        // n_ctx = 20 -> 2 blocks * 16 = seq_capacity 32 (non-exact n_ctx vs block padding).
        // Add an extra 256-byte tail beyond total_bytes to verify view length has zero drift.
        let cfg = ArenaAllocationConfig::new("qwen-layer-major", 20, 4, 3)
            .with_head_count_kv(2)
            .with_vocab_size(64)
            .with_kv_dtype(KV_DTYPE_F32);
        let layout = compute_arena_layout(&cfg).expect("layer-major layout");
        assert_eq!(layout.block_count, 2);
        assert_eq!(layout.seq_capacity, 32);

        let total_usize = usize::try_from(layout.total_bytes).expect("total_bytes fits usize");
        let lines = (total_usize + 256).div_ceil(64);
        let mut backing = vec![AlignedLine([0_u8; 64]); lines];
        let arena_bytes = backing.as_mut_bytes();

        {
            let (hdr_mut, _) =
                ArenaHeader::mut_from_prefix(&mut arena_bytes[..total_usize]).expect("header");
            populate_arena_header(hdr_mut, layout.total_bytes, &cfg, &layout)
                .expect("populate ArenaHeader");
        }

        {
            let (hdr, _) = ArenaHeader::ref_from_prefix(arena_bytes).expect("header");
            assert_eq!(arena_seq_capacity(hdr), 32);
            assert_eq!(arena_kv_dtype(hdr), KV_DTYPE_F32);
            assert_eq!(hdr.n_ctx, 20);
            assert_eq!(hdr.num_layers, 3);
            assert_eq!(hdr.head_count_kv, 2);
            assert_eq!(hdr.head_dim, 4);
            assert_eq!(hdr.vocab_size, 64);
        }

        {
            let mut view = kv_cache_view_from_arena_slice_mut(arena_bytes)
                .expect("construct contiguous layer-major KvCacheView");
            let geom = view.geometry();
            assert_eq!(geom.num_layers, 3);
            assert_eq!(geom.n_heads_kv, 2);
            assert_eq!(geom.head_dim, 4);
            assert_eq!(geom.seq_capacity, 32);
            assert_eq!(view.kv_dim(), 8);

            // Populate positions 0..=20 in every layer with deterministic markers.
            for layer in 0..3 {
                for pos in 0..=20 {
                    let (k_slot, v_slot) = view
                        .write_pos_range(layer, pos)
                        .expect("write within seq_capacity");
                    for d in 0..8 {
                        let code = u16::try_from(layer * 1000 + pos * 10 + d).unwrap();
                        let marker = f32::from(code);
                        k_slot[d] = marker;
                        v_slot[d] = -marker - 0.5;
                    }
                }
            }

            // Read back contiguous 0..=20 range (`end_pos = 21`) across each layer.
            for layer in 0..3 {
                let (k_valid, v_valid) = view
                    .valid_kv_range(layer, 21)
                    .expect("valid range up to pos 20");
                assert_eq!(k_valid.len(), 21 * 8);
                assert_eq!(v_valid.len(), 21 * 8);
                for pos in 0..=20 {
                    let base = pos * 8;
                    for d in 0..8 {
                        let code = u16::try_from(layer * 1000 + pos * 10 + d).unwrap();
                        let expected_k = f32::from(code);
                        let expected_v = -expected_k - 0.5;
                        assert_eq!(k_valid[base + d], expected_k);
                        assert_eq!(v_valid[base + d], expected_v);
                    }
                }

                // Remaining positions 21..32 in plane stay zeroed without bleed.
                let (k_plane, v_plane) = view.layer_kv(layer).expect("layer_kv");
                assert_eq!(k_plane.len(), 32 * 8);
                assert!(k_plane[21 * 8..].iter().all(|&x| x == 0.0));
                assert!(v_plane[21 * 8..].iter().all(|&x| x == 0.0));
            }
        }

        // Verify fail-closed rejection on non-F32 KV dtype, multi-slot, and MLA flag.
        let (hdr_mut, _) = ArenaHeader::mut_from_prefix(arena_bytes).expect("mut header");
        hdr_mut.set_kv_dtype(KvDtype::F16);
        assert!(kv_cache_view_from_arena_slice_mut(arena_bytes).is_err());

        let (hdr_mut, _) = ArenaHeader::mut_from_prefix(arena_bytes).expect("mut header");
        hdr_mut.set_kv_dtype(KvDtype::F32);
        hdr_mut.active_slots = 2;
        assert!(kv_cache_view_from_arena_slice_mut(arena_bytes).is_err());

        let (hdr_mut, _) = ArenaHeader::mut_from_prefix(arena_bytes).expect("mut header");
        hdr_mut.active_slots = 1;
        hdr_mut._reserved[ARENA_RESERVED_MLA_FLAG_OFFSET] = 1;
        assert!(kv_cache_view_from_arena_slice_mut(arena_bytes).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_handle_real_model_geometry_fd_roundtrip() {
        let holder = KvHolder::new();
        let req = AllocateArenaRequest {
            model_id: "qwen2.5-7b-fd-roundtrip".to_owned(),
            arena_bytes: 16 * 1024 * 1024,
            n_ctx: 20,
            max_blocks: 2,
            num_layers: 28,
            head_count_kv: 8,
            head_dim: 128,
            vocab_size: 128_256,
            kv_dtype: KV_DTYPE_F32,
        };
        let alloc = holder
            .allocate_from_request(&req)
            .expect("real geometry arena allocation");
        let fd = holder.get_arena_fd(alloc.arena_id).expect("arena fd");
        let len = usize::try_from(alloc.total_bytes).expect("arena len fits usize");
        let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "test_kv_view", false)
            .expect("rw arena mapping");

        let mut handle = ArenaHandle::new(&mut rw).expect("ArenaHandle validates header");
        let hdr = handle.header().expect("header");
        assert_eq!(hdr.num_layers, 28);
        assert_eq!(hdr.head_count_kv, 8);
        assert_eq!(hdr.head_dim, 128);
        assert_eq!(hdr.vocab_size, 128_256);
        assert_eq!(hdr.n_ctx, 20);
        assert_eq!(arena_seq_capacity(hdr), 32);

        let mut view = handle
            .kv_cache_view_mut()
            .expect("kv_cache_view_mut returns contiguous view");
        assert_eq!(view.geometry().num_layers, 28);
        assert_eq!(view.geometry().seq_capacity, 32);
        assert_eq!(view.kv_dim(), 1024);

        for pos in 0..=20 {
            let val = f32::from(u16::try_from(pos).unwrap()) + 1.0;
            let (k, v) = view.write_pos_range(27, pos).expect("write layer 27");
            k.fill(val);
            v.fill(-val);
        }
        let (k_valid, v_valid) = view.valid_kv_range(27, 21).expect("read layer 27 0..=20");
        assert_eq!(k_valid.len(), 21 * 1024);
        assert_eq!(v_valid.len(), 21 * 1024);
        assert_eq!(k_valid[20 * 1024], 21.0);
        assert_eq!(v_valid[20 * 1024], -21.0);
    }

    #[repr(C, align(64))]
    #[derive(Clone, Copy, FromBytes, IntoBytes, zerocopy::Immutable)]
    struct AlignedArenaLine([u8; 64]);

    fn build_in_memory_kv_arena(
        n_ctx: u32,
        head_dim: u32,
        num_layers: u32,
        head_count_kv: u32,
    ) -> (
        ArenaAllocationConfig,
        ComputedArenaLayout,
        Vec<AlignedArenaLine>,
    ) {
        let cfg = ArenaAllocationConfig::new("wave3-batched-prefill", n_ctx, head_dim, num_layers)
            .with_head_count_kv(head_count_kv)
            .with_vocab_size(64)
            .with_kv_dtype(KV_DTYPE_F32);
        let layout = compute_arena_layout(&cfg).expect("valid layout");
        let total_usize = usize::try_from(layout.total_bytes).expect("total_bytes fits usize");
        let lines = total_usize.div_ceil(64);
        let mut backing = vec![AlignedArenaLine([0_u8; 64]); lines];
        let arena_bytes = backing.as_mut_bytes();
        {
            let (hdr_mut, _) =
                ArenaHeader::mut_from_prefix(&mut arena_bytes[..total_usize]).expect("header");
            populate_arena_header(hdr_mut, layout.total_bytes, &cfg, &layout)
                .expect("populate ArenaHeader");
        }
        let meta_start = usize::try_from(layout.meta_table_offset).expect("meta_table_offset");
        let entry_size = size_of::<KvBlockMeta>();
        for idx in 0..usize::try_from(layout.block_count).expect("block_count") {
            let block_id = u32::try_from(idx).expect("block_id fits u32");
            let off = meta_start + idx * entry_size;
            let meta = KvBlockMeta::mut_from_bytes(&mut arena_bytes[off..off + entry_size])
                .expect("KvBlockMeta");
            meta.block_id = block_id;
            meta.tag = jero_core::ipc::layout::TAG_VOLATILE;
            meta.refcount = 1;
            meta.committed_token_count = 0;
        }
        (cfg, layout, backing)
    }

    fn make_batch_vectors(
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        kv_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut k = Vec::with_capacity(seq_len * kv_dim);
        let mut v = Vec::with_capacity(seq_len * kv_dim);
        for rel in 0..seq_len {
            let pos = start_pos + rel;
            for d in 0..kv_dim {
                let k_code = u16::try_from(layer * 5000 + pos * 16 + d).expect("k_code fits u16");
                let k_val = f32::from(k_code) + 0.25;
                k.push(k_val);
                v.push(-k_val - 0.5);
            }
        }
        (k, v)
    }

    #[test]
    fn test_batched_append_layer_batch_non_power_of_two_chunks_and_bounds() {
        // n_ctx = 60 -> seq_capacity = 64 (4 blocks of 16), kv_dim = 2 * 4 = 8.
        let (_, _, mut backing) = build_in_memory_kv_arena(60, 4, 2, 2);
        let arena_bytes = backing.as_mut_bytes();

        let mut writer =
            ArenaKvWriter::from_arena_slice(arena_bytes).expect("construct ArenaKvWriter");
        assert_eq!(writer.seq_capacity(), 64);
        assert_eq!(writer.kv_dim(), 8);
        assert_eq!(writer.block_count(), 4);

        let chunk_lens = [1_usize, 7, 16, 33]; // total = 57 tokens
        for layer in 0..2 {
            let mut cursor = 0_usize;
            for &seq_len in &chunk_lens {
                let (k_batch, v_batch) = make_batch_vectors(layer, cursor, seq_len, 8);
                let end_pos = cursor + seq_len;
                let (k_prefix, v_prefix) = writer
                    .append_layer_batch(layer, cursor, &k_batch, &v_batch)
                    .expect("append chunk");
                assert_eq!(k_prefix.len(), end_pos * 8);
                assert_eq!(v_prefix.len(), end_pos * 8);

                // Verify entire 0..end_pos prefix matches deterministic marker values.
                let (expected_k, expected_v) = make_batch_vectors(layer, 0, end_pos, 8);
                assert_eq!(k_prefix, expected_k.as_slice());
                assert_eq!(v_prefix, expected_v.as_slice());

                cursor = end_pos;
            }
            assert_eq!(cursor, 57);
        }

        // Fail-closed boundary paths before mutating any state:
        let (k_8, v_8) = make_batch_vectors(0, 57, 8, 8);
        assert!(matches!(
            writer.append_layer_batch(0, 57, &k_8, &v_8),
            Err(KvError::InvalidDimensions(_))
        )); // 57 + 8 = 65 > seq_capacity (64)
        assert!(matches!(
            writer.append_layer_batch(2, 0, &k_8, &v_8),
            Err(KvError::InvalidDimensions(_))
        )); // layer 2 out of bounds
        assert!(matches!(
            writer.append_layer_batch(0, 57, &k_8, &v_8[..v_8.len() - 8]),
            Err(KvError::InvalidDimensions(_))
        )); // mismatched k_batch / v_batch lengths
        assert!(matches!(
            writer.append_layer_batch(0, 57, &k_8[..7], &v_8[..7]),
            Err(KvError::InvalidDimensions(_))
        )); // non-multiple of kv_dim
        assert!(matches!(
            writer.append_layer_batch(0, usize::MAX, &k_8, &v_8),
            Err(KvError::InvalidDimensions(_))
        )); // overflow guard

        // Ensure trailing positions [57..64) stayed zeroed after failed overflow append.
        let reader = ArenaKvReader::from_arena_slice(backing.as_bytes()).expect("ArenaKvReader");
        for layer in 0..2 {
            let (k_tail, v_tail) = reader.layer_kv_range(layer, 57, 64).expect("tail range");
            assert!(k_tail.iter().all(|&x| x == 0.0));
            assert!(v_tail.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_commit_token_span_across_block_boundaries_miri() {
        // Part 1: exact progression 0..15, 15..16, 16..40 across 3 16-token blocks (seq_capacity = 48).
        {
            let (_, _, mut backing) = build_in_memory_kv_arena(48, 4, 2, 1);
            {
                let mut writer =
                    ArenaKvWriter::from_arena_slice(backing.as_mut_bytes()).expect("writer");
                assert_eq!(writer.block_count(), 3);
                assert_eq!(writer.committed_tokens().expect("initial watermark"), 0);

                // 0..15 -> [15, 0, 0]
                writer.commit_token_span(0, 15).expect("commit 0..15");
                assert_eq!(writer.block_committed_tokens(0).unwrap(), 15);
                assert_eq!(writer.block_committed_tokens(1).unwrap(), 0);
                assert_eq!(writer.block_committed_tokens(2).unwrap(), 0);
                assert_eq!(writer.committed_tokens().unwrap(), 15);

                // Reject gaps, overlaps, and out-of-bounds spans without mutating blocks.
                assert!(writer.commit_token_span(16, 1).is_err());
                assert!(writer.commit_token_span(14, 2).is_err());
                assert!(writer.commit_token_span(0, 15).is_err());
                assert_eq!(writer.block_committed_tokens(0).unwrap(), 15);

                // 15..16 -> [16, 0, 0]
                writer.commit_token_span(15, 1).expect("commit 15..16");
                assert_eq!(writer.block_committed_tokens(0).unwrap(), 16);
                assert_eq!(writer.block_committed_tokens(1).unwrap(), 0);
                assert_eq!(writer.block_committed_tokens(2).unwrap(), 0);
                assert_eq!(writer.committed_tokens().unwrap(), 16);

                // 16..40 -> [16, 16, 8]
                writer.commit_token_span(16, 24).expect("commit 16..40");
                assert_eq!(writer.block_committed_tokens(0).unwrap(), 16);
                assert_eq!(writer.block_committed_tokens(1).unwrap(), 16);
                assert_eq!(writer.block_committed_tokens(2).unwrap(), 8);
                assert_eq!(writer.committed_tokens().unwrap(), 40);

                // Out-of-bounds beyond seq_capacity=48 fails closed.
                assert!(writer.commit_token_span(40, 9).is_err());
                assert_eq!(writer.committed_tokens().unwrap(), 40);
            }

            let reader = ArenaKvReader::from_arena_slice(backing.as_bytes()).expect("reader");
            assert_eq!(reader.block_committed_tokens(0).unwrap(), 16);
            assert_eq!(reader.block_committed_tokens(1).unwrap(), 16);
            assert_eq!(reader.block_committed_tokens(2).unwrap(), 8);
            assert_eq!(reader.committed_tokens().unwrap(), 40);
            assert!(reader.block_committed_tokens(3).is_err());
        }

        // Part 2: chunked prefill 0..128, 128..256, 256..300 (n_ctx = 300 -> seq_capacity = 304, 19 blocks).
        {
            let (_, _, mut backing) = build_in_memory_kv_arena(300, 2, 2, 2);
            let chunks = [(0_usize, 128_usize), (128, 128), (256, 44)];
            for &(start_pos, seq_len) in &chunks {
                {
                    let mut writer =
                        ArenaKvWriter::from_arena_slice(backing.as_mut_bytes()).expect("writer");
                    for layer in 0..2 {
                        let (k_batch, v_batch) = make_batch_vectors(layer, start_pos, seq_len, 4);
                        let (k_pref, _) = writer
                            .append_layer_batch(layer, start_pos, &k_batch, &v_batch)
                            .expect("append chunk");
                        assert_eq!(k_pref.len(), (start_pos + seq_len) * 4);
                    }
                    writer
                        .commit_prefill_range(start_pos, seq_len)
                        .expect("commit prefill chunk");
                }
                let reader = ArenaKvReader::from_arena_slice(backing.as_bytes()).expect("reader");
                assert_eq!(reader.committed_tokens().unwrap(), start_pos + seq_len);
            }

            let reader = ArenaKvReader::from_arena_slice(backing.as_bytes()).expect("reader");
            assert_eq!(reader.block_count(), 19);
            assert_eq!(reader.committed_tokens().unwrap(), 300);
            for b in 0..18 {
                assert_eq!(reader.block_committed_tokens(b).unwrap(), 16);
            }
            assert_eq!(reader.block_committed_tokens(18).unwrap(), 12);

            for layer in 0..2 {
                let (k_commit, v_commit) = reader
                    .committed_kv_range(layer)
                    .expect("committed_kv_range");
                let (expected_k, expected_v) = make_batch_vectors(layer, 0, 300, 4);
                assert_eq!(k_commit, expected_k.as_slice());
                assert_eq!(v_commit, expected_v.as_slice());
            }
        }

        // Part 3: ArenaKvWriter::from_parts & corrupted non-contiguous metadata rejection.
        {
            let geom = KvGeometry::new(1, 1, 4, 32, None).expect("geom");
            let mut kv_buf = vec![0.0_f32; geom.required_f32_len().unwrap()];
            let mut metas = [KvBlockMeta::new(0, 15), KvBlockMeta::new(1, 2)];
            let view = KvCacheView::new(&mut kv_buf, geom).expect("view");
            let mut writer = ArenaKvWriter::from_parts(view, &mut metas).expect("from_parts");
            assert!(matches!(
                writer.committed_tokens(),
                Err(KvError::InvalidDimensions(_))
            ));
            assert!(matches!(
                writer.commit_token_span(17, 1),
                Err(KvError::InvalidDimensions(_))
            ));
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_handle_batched_prefill_roundtrip() {
        let holder = KvHolder::new();
        let cfg = ArenaAllocationConfig::new("qwen-wave3-prefill", 300, 4, 2)
            .with_head_count_kv(2)
            .with_vocab_size(128)
            .with_kv_dtype(KV_DTYPE_F32);
        let alloc = holder.allocate(&cfg).expect("allocate wave3 arena");

        let writable_fd = holder.get_arena_fd(alloc.arena_id).expect("writable fd");
        let readonly_fd = holder.get_arena_fd(alloc.arena_id).expect("readonly fd");
        let len = usize::try_from(alloc.total_bytes).expect("arena len fits usize");

        let mut rw = SharedMemoryArena::from_existing_fd(writable_fd, len, "test_wave3_rw", false)
            .expect("rw mapping");
        let mut ro = SharedMemoryArena::from_existing_fd(readonly_fd, len, "test_wave3_ro", true)
            .expect("ro mapping");

        // Read-only arena rejects mutable writer cleanly without panicking.
        assert!(matches!(
            ArenaKvWriter::from_arena(&mut ro),
            Err(KvError::IpcError(_))
        ));

        let mut handle = ArenaHandle::new(&mut rw).expect("ArenaHandle");
        let chunks = [(0_usize, 128_usize), (128, 128), (256, 44)];
        for &(start_pos, seq_len) in &chunks {
            {
                let mut writer = handle.kv_writer().expect("kv_writer");
                for layer in 0..2 {
                    let (k_batch, v_batch) = make_batch_vectors(layer, start_pos, seq_len, 8);
                    let (k_pref, v_pref) = writer
                        .append_layer_batch(layer, start_pos, &k_batch, &v_batch)
                        .expect("append chunk via ArenaHandle");
                    assert_eq!(k_pref.len(), (start_pos + seq_len) * 8);
                    assert_eq!(v_pref.len(), (start_pos + seq_len) * 8);
                }
            }
            handle
                .commit_prefill_range(start_pos, seq_len)
                .expect("commit_prefill_range");
            assert_eq!(
                handle.committed_tokens().expect("committed_tokens"),
                start_pos + seq_len
            );
        }

        let ro_reader = ArenaKvReader::from_arena(&ro).expect("ro ArenaKvReader");
        assert_eq!(ro_reader.committed_tokens().unwrap(), 300);
        for b in 0..18 {
            assert_eq!(ro_reader.block_committed_tokens(b).unwrap(), 16);
        }
        assert_eq!(ro_reader.block_committed_tokens(18).unwrap(), 12);
        let (k_all, v_all) = ro_reader.committed_kv_range(1).expect("committed layer 1");
        let (expected_k, expected_v) = make_batch_vectors(1, 0, 300, 8);
        assert_eq!(k_all, expected_k.as_slice());
        assert_eq!(v_all, expected_v.as_slice());
    }

    #[repr(C, align(128))]
    #[derive(Clone, Copy, FromBytes, IntoBytes, zerocopy::Immutable)]
    struct AlignedVectorLine([u8; VECTOR_PLANE_ALIGNMENT]);

    fn init_aligned_slice_arena(cfg: &ArenaAllocationConfig) -> Vec<AlignedVectorLine> {
        let layout = compute_arena_layout(cfg).expect("layout");
        let total = usize::try_from(layout.total_bytes).expect("usize");
        let lines = total.div_ceil(VECTOR_PLANE_ALIGNMENT);
        let mut backing = vec![AlignedVectorLine([0_u8; VECTOR_PLANE_ALIGNMENT]); lines];
        let arena_bytes = &mut backing.as_mut_bytes()[..total];
        let (hdr_bytes, _) = arena_bytes.split_at_mut(size_of::<ArenaHeader>());
        let hdr = ArenaHeader::mut_from_bytes(hdr_bytes).expect("hdr");
        populate_arena_header(hdr, layout.total_bytes, cfg, &layout).expect("populate");
        let meta_start = usize::try_from(layout.meta_table_offset).expect("meta");
        for b in 0..layout.block_count as usize {
            let off = meta_start + b * size_of::<KvBlockMeta>();
            let m =
                KvBlockMeta::mut_from_bytes(&mut arena_bytes[off..off + size_of::<KvBlockMeta>()])
                    .expect("meta");
            *m = KvBlockMeta::default();
        }
        backing
    }

    #[test]
    fn test_vector_plane_128b_alignment_across_dtypes_and_hybrid_miri() {
        let align = VECTOR_PLANE_ALIGNMENT as u64;
        // Non-128B raw Q8_0 plane: 16 rows * 34 B = 544 B -> padded stride = 640 B.
        for &(dtype, head_dim, expected_row, expected_stride) in &[
            (KV_DTYPE_F32, 32_u32, 128_u64, 2048_u64),
            (KV_DTYPE_F16, 16_u32, 32_u64, 512_u64),
            (KV_DTYPE_Q8_0, 32_u32, 34_u64, 640_u64),
        ] {
            let cfg = ArenaAllocationConfig::new("align-check", 16, head_dim, 2)
                .with_head_count_kv(1)
                .with_vocab_size(64)
                .with_kv_dtype(dtype);
            let layout = compute_arena_layout(&cfg).expect("aligned layout");
            assert_eq!(layout.token_slots_offset % align, 0);
            assert_eq!(layout.kv_blocks_offset % align, 0);
            assert_eq!(layout.output_slabs_offset % align, 0);
            assert_eq!(layout.kv_row_bytes, expected_row);
            assert_eq!(layout.kv_plane_stride_bytes, expected_stride);
            assert_eq!(layout.kv_plane_stride_bytes % align, 0);
            for layer in 0..2 {
                assert_eq!(layout.k_plane_offset(layer) % align, 0);
                assert_eq!(layout.v_plane_offset(layer) % align, 0);
            }
        }

        // Hybrid sliding-window + global policies preserve 128-byte plane alignment.
        let hybrid_cfg = ArenaAllocationConfig::new("hybrid-align", 32, 32, 2)
            .with_head_count_kv(1)
            .with_vocab_size(64)
            .with_kv_dtype(KV_DTYPE_Q8_0)
            .with_layer_policies(vec![
                LayerKvPolicy::SlidingWindow { window_tokens: 16 },
                LayerKvPolicy::GlobalFullContext,
            ]);
        let hybrid = compute_arena_layout(&hybrid_cfg).expect("hybrid layout");
        assert_eq!(hybrid.kv_blocks_offset % align, 0);
        assert_eq!(hybrid.output_slabs_offset % align, 0);
        assert_eq!(
            hybrid.output_slabs_offset - hybrid.kv_blocks_offset,
            // Layer 0 (16 tok = 544 -> 640 * 2) + Layer 1 (32 tok = 1088 -> 1152 * 2)
            (640 + 1152) * 2
        );

        // Runtime K/V plane pointers inside a 128B-aligned Q8_0 slab are 128B-aligned.
        let q8_cfg = ArenaAllocationConfig::new("q8-ptr", 16, 32, 2)
            .with_head_count_kv(1)
            .with_vocab_size(32)
            .with_kv_dtype(KV_DTYPE_Q8_0);
        let mut backing = init_aligned_slice_arena(&q8_cfg);
        let arena_bytes = backing.as_mut_bytes();
        let reader = ArenaKvReader::from_arena_slice(arena_bytes).expect("Q8_0 reader");
        assert_eq!(reader.kv_dtype(), KvDtype::Q8_0);
        assert_eq!(reader.kv_row_bytes(), 34);
        assert_eq!(reader.kv_plane_stride_bytes(), 640);
        for layer in 0..2 {
            let (k_plane, v_plane) = reader.layer_kv_bytes(layer).expect("layer bytes");
            assert_eq!(k_plane.len(), 544);
            assert_eq!(v_plane.len(), 544);
            assert_eq!((k_plane.as_ptr() as usize) % VECTOR_PLANE_ALIGNMENT, 0);
            assert_eq!((v_plane.as_ptr() as usize) % VECTOR_PLANE_ALIGNMENT, 0);
        }
    }

    fn make_unit_batch(
        layer: usize,
        start_pos: usize,
        seq_len: usize,
        kv_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut k = Vec::with_capacity(seq_len * kv_dim);
        let mut v = Vec::with_capacity(seq_len * kv_dim);
        for rel in 0..seq_len {
            let pos = start_pos + rel;
            for d in 0..kv_dim {
                let bucket = u8::try_from((layer * 19 + pos * 11 + d * 5) % 64).expect("< 64");
                let val = f32::from(bucket) / 31.5 - 1.0;
                k.push(val * 0.9);
                v.push(-val * 0.8 + 0.05);
            }
        }
        (k, v)
    }

    #[test]
    fn test_quantized_kv_writer_reader_and_scratch_roundtrip_miri() {
        use jero_core::ScratchWorkspace;

        for &(dtype, kv_dim, max_tol, min_cos) in &[
            (KV_DTYPE_F16, 16_usize, 1e-3_f32, 0.999_99_f64),
            (KV_DTYPE_Q8_0, 32_usize, 1.5e-2_f32, 0.999_f64),
        ] {
            let cfg =
                ArenaAllocationConfig::new("quant-miri", 16, u32::try_from(kv_dim).unwrap(), 2)
                    .with_head_count_kv(1)
                    .with_vocab_size(32)
                    .with_kv_dtype(dtype);
            let layout = compute_arena_layout(&cfg).expect("layout");
            let total = usize::try_from(layout.total_bytes).expect("usize");
            let mut backing = init_aligned_slice_arena(&cfg);
            let arena_bytes = &mut backing.as_mut_bytes()[..total];

            {
                let mut writer = ArenaKvWriter::from_arena_slice(arena_bytes).expect("writer");
                assert!(writer.kv_cache_view().is_none());
                assert_eq!(writer.kv_dtype(), KvDtype::from_u8(dtype).unwrap());
                for layer in 0..2 {
                    let (k0, v0) = make_unit_batch(layer, 0, 6, kv_dim);
                    assert_eq!(
                        writer
                            .append_quantized_kv_batch(layer, 0, &k0, &v0)
                            .expect("prefill"),
                        6
                    );
                    let (k1, v1) = make_unit_batch(layer, 6, 4, kv_dim);
                    assert_eq!(
                        writer
                            .append_quantized_kv_batch(layer, 6, &k1, &v1)
                            .expect("decode"),
                        10
                    );
                }
                writer.commit_token_span(0, 6).expect("commit prefill");
                writer.commit_token_span(6, 4).expect("commit decode");
            }

            let reader = ArenaKvReader::from_arena_slice(arena_bytes).expect("reader");
            assert_eq!(reader.committed_tokens().expect("committed"), 10);
            let mut scratch = ScratchWorkspace::new(32 * 1024).expect("scratch workspace");
            let mut k_buf = vec![0.0_f32; 10 * kv_dim];
            let mut v_buf = vec![0.0_f32; 10 * kv_dim];

            for layer in 0..2 {
                let (expected_k, expected_v) = make_unit_batch(layer, 0, 10, kv_dim);
                let (rk, rv) = reader
                    .read_committed_kv_range_f32(layer, &mut k_buf, &mut v_buf)
                    .expect("read_committed_kv_range_f32");
                let scope = scratch.scope();
                let (sk, sv) = reader
                    .read_committed_kv_range_in_scratch(layer, &scope)
                    .expect("read_committed_kv_range_in_scratch");
                assert_eq!((sk.as_ptr() as usize) % VECTOR_PLANE_ALIGNMENT, 0);
                assert_eq!((sv.as_ptr() as usize) % VECTOR_PLANE_ALIGNMENT, 0);
                assert_eq!(rk, sk);
                assert_eq!(rv, sv);

                let mut dot = 0.0_f64;
                let mut n_a = 0.0_f64;
                let mut n_b = 0.0_f64;
                for (&exp, &got) in expected_k
                    .iter()
                    .chain(expected_v.iter())
                    .zip(sk.iter().chain(sv.iter()))
                {
                    assert!(
                        (exp - got).abs() <= max_tol,
                        "dtype={dtype:?} diff {} > {max_tol}",
                        (exp - got).abs()
                    );
                    dot += f64::from(exp) * f64::from(got);
                    n_a += f64::from(exp) * f64::from(exp);
                    n_b += f64::from(got) * f64::from(got);
                }
                let cosine = dot / (n_a.sqrt() * n_b.sqrt());
                assert!(cosine >= min_cos, "cosine {cosine} < {min_cos}");
            }
        }

        // Q8_0 requires kv_dim divisible by 32.
        let bad_q8 = ArenaAllocationConfig::new("bad-q8", 16, 16, 1)
            .with_head_count_kv(1)
            .with_kv_dtype(KV_DTYPE_Q8_0);
        assert!(matches!(
            compute_arena_layout(&bad_q8),
            Err(KvError::InvalidDimensions(_))
        ));
    }

    #[test]
    fn test_recurrent_memlock_o1_independence_miri() {
        let cfg_short = ArenaAllocationConfig::new("ssm-o1", 64, 64, 0)
            .with_vocab_size(64)
            .with_recurrent_state(4, 128, 512);
        let cfg_long = ArenaAllocationConfig::new("ssm-o1", 65_536, 64, 0)
            .with_vocab_size(64)
            .with_recurrent_state(4, 128, 512);

        let est_short = cfg_short
            .estimate_recurrent_state_bytes()
            .expect("estimate recurrent short");
        let layout_short = compute_arena_layout(&cfg_short).expect("layout short");
        let layout_long = compute_arena_layout(&cfg_long).expect("layout long");

        assert_eq!(est_short, layout_short.recurrent_total_bytes);
        assert_eq!(
            layout_short.recurrent_total_bytes,
            layout_long.recurrent_total_bytes
        );
        assert_eq!(layout_short.total_bytes, layout_long.total_bytes);

        let holder = KvHolder::with_memlock_ceiling(layout_short.total_bytes + 4096);
        assert!(holder.available_lockable_bytes() >= layout_long.total_bytes);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_arena_handle_recurrent_ping_pong_roundtrip() {
        let holder = KvHolder::new();
        let cfg = ArenaAllocationConfig::new("mamba-handle", 65_536, 64, 0)
            .with_vocab_size(128)
            .with_recurrent_state(4, 128, 512);
        let alloc = holder
            .allocate(&cfg)
            .expect("allocate O(1) recurrent arena");
        let fd = holder.get_arena_fd(alloc.arena_id).expect("fd");
        let len = usize::try_from(alloc.total_bytes).expect("usize");
        let mut rw =
            SharedMemoryArena::from_existing_fd(fd, len, "test_handle_rec", false).expect("rw");
        let mut handle = ArenaHandle::new(&mut rw).expect("ArenaHandle");

        let hdr0 = handle.recurrent_header().expect("recurrent_header");
        assert_eq!(hdr0.num_recurrent_layers, 4);
        assert_eq!(hdr0.committed_step, 0);

        {
            let (_h, active, inactive) = handle.recurrent_banks_mut().expect("banks");
            assert!(active.iter().all(|&b| b == 0));
            inactive.fill(0x5A);
        }
        let hdr1 = handle.commit_recurrent_step(1).expect("commit step 1");
        assert_eq!(hdr1.committed_step, 1);
        assert_eq!(handle.recurrent_header().expect("hdr1").committed_step, 1);

        {
            let (_h, active, _inactive) = handle.recurrent_banks_mut().expect("banks post-commit");
            assert!(active.iter().all(|&b| b == 0x5A));
        }
    }
}
