//! Context memory holder daemon and shared memory arena management for the `SpacemiT` K3 AI platform.

pub use jero_core::ipc::layout::estimate_recurrent_state_bytes;
pub use jero_core::{DoryKvCacheMode, DoryLayerType, DoryMetadata, KvDtype};
use jero_core::{K3Error, ModelMetadata};

pub const KV_DAEMON_NAME: &str = "jero-kv-cache";

/// Canonical 72-layer disaggregated sub-layer pattern for `Dory1.0-4B`.
pub const CANONICAL_DORY_4B_PATTERN: &str =
    "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-";

pub mod holder;
pub mod quant_kv;
pub mod server;

pub use holder::{
    ArenaAllocationConfig, ArenaAllocationResult, ArenaHandle, ArenaKvReader, ArenaKvWriter,
    ComputedArenaLayout, KV_BLOCK_TOKEN_CAPACITY, KvError, KvHolder, KvHolderError, LayerKvPolicy,
    arena_kv_dtype, arena_seq_capacity, commit_recurrent_step_in_arena,
    commit_recurrent_step_in_region, compute_arena_layout, compute_hybrid_kv_block_count,
    kv_cache_view_from_arena_slice_mut, read_recurrent_slot_header_from_region,
    recurrent_banks_from_region_mut, recurrent_banks_in_arena_mut, recurrent_region_offset,
};
pub use quant_kv::{
    VECTOR_PLANE_ALIGNMENT, dequantize_q8_0_block, f16_to_f32, f32_to_f16, pack_kv_batch_plane,
    pack_kv_row, quantize_q8_0_block, unpack_kv_range_plane, unpack_kv_row,
};
pub use server::{
    DEFAULT_SOCKET_PATH, KvServer, ServerConfig, SocketFileGuard, handle_connection,
    prepare_and_bind_socket,
};

/// Maps Dory1.0 virtual execution sub-layer indices (`0 .. n_input + n_recurrent * n_loops + n_output`)
/// to dense attention-only KV cache slot indices (`0 .. active_attention_kv_layers()`).
///
/// Pure MLP sub-layers (`'-'`) have zero KV heads and always resolve to `None`.
/// In [`DoryKvCacheMode::PerLoop`] mode, each recurrent loop iteration receives distinct KV cache
/// slots (`36` for $K=1$, `56` for $K=2$, `96` for $K=4$ on `Dory1.0-4B`).
/// In [`DoryKvCacheMode::LastLoop`] mode, all recurrent loop iterations collapse onto the shared
/// physical recurrent attention slots (`36` active attention KV slots regardless of $K$).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoryVirtualKvMapper {
    n_input_layers: u32,
    n_recurrent_layers: u32,
    n_output_layers: u32,
    n_loops: u32,
    mode: DoryKvCacheMode,
    layer_types: Vec<DoryLayerType>,
    virtual_to_slot: Vec<Option<usize>>,
    active_attention_slots: usize,
}

impl DoryVirtualKvMapper {
    /// Constructs a [`DoryVirtualKvMapper`] from explicit segment counts, loop budget, KV cache mode,
    /// and disaggregated `layer_pattern` (`'S'`, `'*'`, `'-'`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `n_loops` is outside `1..=4`, if total physical
    /// layers is zero, if `layer_pattern` length does not match `n_input + n_recurrent + n_output`,
    /// or if `layer_pattern` contains invalid characters.
    #[expect(
        clippy::too_many_lines,
        reason = "validates 3-segment Dory loop bounds and builds virtual-to-slot table"
    )]
    pub fn new(
        n_input_layers: u32,
        n_recurrent_layers: u32,
        n_output_layers: u32,
        n_loops: u32,
        mode: DoryKvCacheMode,
        layer_pattern: &str,
    ) -> Result<Self, K3Error> {
        if !(1..=4).contains(&n_loops) {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!("DoryVirtualKvMapper requires n_loops in 1..=4, got {n_loops}"),
            });
        }

        let n_physical = n_input_layers
            .saturating_add(n_recurrent_layers)
            .saturating_add(n_output_layers);
        if n_physical == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "DoryVirtualKvMapper requires at least one physical layer".to_owned(),
            });
        }

        let mut layer_types = Vec::with_capacity(n_physical as usize);
        for ch in layer_pattern.chars() {
            let kind = match ch {
                'S' => DoryLayerType::SlidingWindowAttention,
                '*' => DoryLayerType::FullAttention,
                '-' => DoryLayerType::Mlp,
                _ => {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "dory".to_owned(),
                        reason: format!(
                            "invalid character '{ch}' in dory.layer_pattern (expected 'S', '*', or '-')"
                        ),
                    });
                }
            };
            layer_types.push(kind);
        }

        if layer_types.len() != n_physical as usize {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "dory.layer_pattern length ({}) does not match physical layer count ({n_physical})",
                    layer_types.len()
                ),
            });
        }

        let n_virtual = n_input_layers
            .saturating_add(n_recurrent_layers.saturating_mul(n_loops))
            .saturating_add(n_output_layers);

        let mut virtual_to_slot = Vec::with_capacity(n_virtual as usize);
        let mut next_slot = 0usize;

        let is_attn = |kind: DoryLayerType| {
            matches!(
                kind,
                DoryLayerType::SlidingWindowAttention | DoryLayerType::FullAttention
            )
        };

        // 1. Input segment: 0 .. n_input_layers
        for p in 0..n_input_layers {
            if is_attn(layer_types[p as usize]) {
                virtual_to_slot.push(Some(next_slot));
                next_slot += 1;
            } else {
                virtual_to_slot.push(None);
            }
        }

        // 2. Shared recurrent segment: n_loops iterations of n_input_layers .. n_input_layers + n_recurrent_layers
        let rec_base_slot = next_slot;
        let mut rec_attn_count_per_loop = 0usize;
        for loop_idx in 0..n_loops {
            let mut loop_attn_offset = 0usize;
            for offset in 0..n_recurrent_layers {
                let p = n_input_layers + offset;
                if is_attn(layer_types[p as usize]) {
                    let slot = match mode {
                        DoryKvCacheMode::PerLoop => {
                            let s = next_slot;
                            next_slot += 1;
                            s
                        }
                        DoryKvCacheMode::LastLoop => rec_base_slot + loop_attn_offset,
                    };
                    virtual_to_slot.push(Some(slot));
                    loop_attn_offset += 1;
                } else {
                    virtual_to_slot.push(None);
                }
            }
            if loop_idx == 0 {
                rec_attn_count_per_loop = loop_attn_offset;
            }
        }
        if mode == DoryKvCacheMode::LastLoop {
            next_slot = rec_base_slot + rec_attn_count_per_loop;
        }

        // 3. Output segment: n_input_layers + n_recurrent_layers .. n_physical
        for p in (n_input_layers + n_recurrent_layers)..n_physical {
            if is_attn(layer_types[p as usize]) {
                virtual_to_slot.push(Some(next_slot));
                next_slot += 1;
            } else {
                virtual_to_slot.push(None);
            }
        }

        Ok(Self {
            n_input_layers,
            n_recurrent_layers,
            n_output_layers,
            n_loops,
            mode,
            layer_types,
            virtual_to_slot,
            active_attention_slots: next_slot,
        })
    }

    /// Constructs a [`DoryVirtualKvMapper`] for the canonical `Dory1.0-4B` topology (`16 + 40 * K + 16`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `n_loops` is outside `1..=4`.
    pub fn canonical_dory_4b(n_loops: u32, mode: DoryKvCacheMode) -> Result<Self, K3Error> {
        Self::new(16, 40, 16, n_loops, mode, CANONICAL_DORY_4B_PATTERN)
    }

    /// Constructs a [`DoryVirtualKvMapper`] from parsed [`DoryMetadata`], honoring GGUF defaults
    /// unless explicitly overridden by `loop_override` or `mode_override`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `loop_override` is outside `1..=4`.
    pub fn from_dory_metadata(
        dory: &DoryMetadata,
        loop_override: Option<u32>,
        mode_override: Option<DoryKvCacheMode>,
    ) -> Result<Self, K3Error> {
        let n_loops = loop_override.unwrap_or_else(|| dory.num_loops());
        let mode = mode_override.unwrap_or_else(|| dory.recurrent_kv_cache_mode());
        Self::new(
            dory.input_layer_count(),
            dory.recurrent_layer_count(),
            dory.output_layer_count(),
            n_loops,
            mode,
            dory.layer_pattern(),
        )
    }

    /// Constructs a [`DoryVirtualKvMapper`] from [`ModelMetadata`], strictly honoring
    /// `dory.recurrent_kv_cache_mode` and `dory.num_loops` from GGUF metadata unless
    /// explicitly overridden by `loop_override` or `mode_override`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `metadata` does not contain Dory metadata,
    /// if `loop_override` is outside `1..=4`, or if `mode_override` is neither `"per_loop"` nor `"last_loop"`.
    pub fn from_model_metadata(
        metadata: &ModelMetadata,
        loop_override: Option<u32>,
        mode_override: Option<&str>,
    ) -> Result<Self, K3Error> {
        let dory = metadata.dory().ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: metadata.architecture().to_owned(),
            reason: "ModelMetadata does not contain parsed dory.* metadata".to_owned(),
        })?;
        let parsed_mode = match mode_override {
            Some(raw) => Some(DoryKvCacheMode::parse_mode(raw)?),
            None => None,
        };
        Self::from_dory_metadata(dory, loop_override, parsed_mode)
    }

    /// Resolves a virtual execution sub-layer index (`0 .. virtual_layer_count()`) to its dense
    /// attention-only KV cache slot index (`0 .. active_attention_kv_layers()`).
    ///
    /// Returns `None` if `virtual_layer` corresponds to a pure MLP sub-layer (`'-'`) or is out of bounds.
    #[must_use]
    pub fn resolve_kv_slot(&self, virtual_layer: u32) -> Option<usize> {
        self.virtual_to_slot
            .get(virtual_layer as usize)
            .copied()
            .flatten()
    }

    /// Returns the total number of active attention KV cache layer slots allocated under the
    /// configured loop count and [`DoryKvCacheMode`].
    #[must_use]
    pub const fn active_attention_kv_layers(&self) -> usize {
        self.active_attention_slots
    }

    /// Returns the configured recurrent loop count (`1..=4`).
    #[must_use]
    pub const fn n_loops(&self) -> u32 {
        self.n_loops
    }

    /// Returns the active [`DoryKvCacheMode`] (`PerLoop` or `LastLoop`).
    #[must_use]
    pub const fn mode(&self) -> DoryKvCacheMode {
        self.mode
    }

    /// Returns the total number of virtual execution steps (`n_input + n_recurrent * n_loops + n_output`).
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded virtual layer count <= 192"
    )]
    pub fn virtual_layer_count(&self) -> u32 {
        self.virtual_to_slot.len() as u32
    }

    /// Returns the total number of physical weight sub-layers (`n_input + n_recurrent + n_output`).
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded physical layer count <= 72"
    )]
    pub fn physical_layer_count(&self) -> u32 {
        self.layer_types.len() as u32
    }

    /// Maps a virtual execution step (`0 .. virtual_layer_count()`) to its physical sub-layer index (`0 .. physical_layer_count()`).
    #[must_use]
    pub fn physical_layer_for_virtual(&self, virtual_layer: u32) -> Option<u32> {
        if virtual_layer >= self.virtual_layer_count() {
            return None;
        }
        if virtual_layer < self.n_input_layers {
            Some(virtual_layer)
        } else {
            let rec_span = self.n_recurrent_layers.saturating_mul(self.n_loops);
            let rec_end = self.n_input_layers.saturating_add(rec_span);
            if virtual_layer < rec_end {
                let offset = (virtual_layer - self.n_input_layers) % self.n_recurrent_layers;
                Some(self.n_input_layers + offset)
            } else {
                let out_offset = virtual_layer - rec_end;
                Some(self.n_input_layers + self.n_recurrent_layers + out_offset)
            }
        }
    }

    /// Returns the [`DoryLayerType`] (`SlidingWindowAttention`, `FullAttention`, or `Mlp`) for `virtual_layer`.
    #[must_use]
    pub fn layer_type_for_virtual(&self, virtual_layer: u32) -> Option<DoryLayerType> {
        let p = self.physical_layer_for_virtual(virtual_layer)?;
        self.layer_types.get(p as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jero_core::{GgufFile, GgufValue};

    #[test]
    fn test_daemon_identity() {
        assert_eq!(KV_DAEMON_NAME, "jero-kv-cache");
    }

    #[test]
    fn test_dory_virtual_kv_mapper() {
        // 1. PerLoop mode across K = 1, 2, 4
        let mapper_k1 = DoryVirtualKvMapper::canonical_dory_4b(1, DoryKvCacheMode::PerLoop)
            .expect("canonical K=1 PerLoop");
        assert_eq!(mapper_k1.virtual_layer_count(), 72);
        assert_eq!(mapper_k1.active_attention_kv_layers(), 36);

        let mapper_k2 = DoryVirtualKvMapper::canonical_dory_4b(2, DoryKvCacheMode::PerLoop)
            .expect("canonical K=2 PerLoop");
        assert_eq!(mapper_k2.virtual_layer_count(), 112);
        assert_eq!(mapper_k2.active_attention_kv_layers(), 56);

        let mapper_k4 = DoryVirtualKvMapper::canonical_dory_4b(4, DoryKvCacheMode::PerLoop)
            .expect("canonical K=4 PerLoop");
        assert_eq!(mapper_k4.virtual_layer_count(), 192);
        assert_eq!(mapper_k4.active_attention_kv_layers(), 96);

        // Verify MLP sub-layers ('-') return None and attention sub-layers ('S', '*') get dense sequential slots
        assert_eq!(mapper_k2.resolve_kv_slot(0), Some(0)); // 'S'
        assert_eq!(mapper_k2.resolve_kv_slot(1), None); // '-'
        assert_eq!(mapper_k2.resolve_kv_slot(2), Some(1)); // 'S'
        assert_eq!(mapper_k2.resolve_kv_slot(6), Some(3)); // '*'
        assert_eq!(mapper_k2.resolve_kv_slot(14), Some(7)); // last input attention layer
        assert_eq!(mapper_k2.resolve_kv_slot(15), None); // last input MLP layer

        // Loop 0 start (virtual 16 -> physical 16 'S') gets slot 8; Loop 1 start (virtual 56 -> physical 16 'S') gets slot 28 in PerLoop
        assert_eq!(mapper_k2.resolve_kv_slot(16), Some(8));
        assert_eq!(mapper_k2.resolve_kv_slot(56), Some(28));
        // Output block start (virtual 96 -> physical 56 'S') gets slot 48 in PerLoop (8 + 20 + 20)
        assert_eq!(mapper_k2.resolve_kv_slot(96), Some(48));
        assert_eq!(mapper_k2.resolve_kv_slot(110), Some(55));
        assert_eq!(mapper_k2.resolve_kv_slot(111), None);
        assert_eq!(mapper_k2.resolve_kv_slot(112), None); // out of bounds

        // 2. LastLoop mode collapses recurrent loop iterations onto shared physical KV slots (36 total for K=1, 2, 4)
        for k in [1_u32, 2, 4] {
            let last_loop = DoryVirtualKvMapper::canonical_dory_4b(k, DoryKvCacheMode::LastLoop)
                .expect("canonical LastLoop");
            assert_eq!(last_loop.active_attention_kv_layers(), 36);

            // Every recurrent loop l in 0..k maps physical recurrent layer 16 ('S') to slot 8
            for l in 0..k {
                let v_start = 16 + l * 40;
                assert_eq!(last_loop.resolve_kv_slot(v_start), Some(8));
                assert_eq!(last_loop.resolve_kv_slot(v_start + 1), None); // '-'
                assert_eq!(last_loop.resolve_kv_slot(v_start + 38), Some(27)); // last recurrent attn
            }

            // Output segment starts at slot 28 regardless of K
            let out_start = 16 + k * 40;
            assert_eq!(last_loop.resolve_kv_slot(out_start), Some(28));
            assert_eq!(last_loop.resolve_kv_slot(out_start + 14), Some(35));
            assert_eq!(last_loop.resolve_kv_slot(out_start + 15), None);
        }
    }

    #[test]
    fn test_dory_virtual_kv_mapper_from_metadata_and_overrides() {
        let extra_kvs = [
            ("dory.block_count", GgufValue::Uint32(0)),
            ("dory.input_layer_count", GgufValue::Uint32(16)),
            ("dory.recurrent_layer_count", GgufValue::Uint32(40)),
            ("dory.output_layer_count", GgufValue::Uint32(16)),
            ("dory.num_loops", GgufValue::Uint32(2)),
            (
                "dory.recurrent_kv_cache_mode",
                GgufValue::String("per_loop"),
            ),
            (
                "dory.layer_pattern",
                GgufValue::String(CANONICAL_DORY_4B_PATTERN),
            ),
        ];
        let bytes =
            jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs("dory", 32, &extra_kvs, &[]);
        let gguf = GgufFile::parse(&bytes).expect("parse synthetic dory gguf");
        let meta = gguf.metadata();

        // Default from GGUF metadata: K=2, per_loop -> 56 slots
        let default_mapper =
            DoryVirtualKvMapper::from_model_metadata(meta, None, None).expect("from metadata");
        assert_eq!(default_mapper.n_loops(), 2);
        assert_eq!(default_mapper.mode(), DoryKvCacheMode::PerLoop);
        assert_eq!(default_mapper.active_attention_kv_layers(), 56);

        // Override to K=4, last_loop -> 36 slots
        let override_mapper =
            DoryVirtualKvMapper::from_model_metadata(meta, Some(4), Some("last_loop"))
                .expect("override mapper");
        assert_eq!(override_mapper.n_loops(), 4);
        assert_eq!(override_mapper.mode(), DoryKvCacheMode::LastLoop);
        assert_eq!(override_mapper.active_attention_kv_layers(), 36);

        // Invalid overrides rejected
        assert!(DoryVirtualKvMapper::from_model_metadata(meta, Some(0), None).is_err());
        assert!(DoryVirtualKvMapper::from_model_metadata(meta, Some(5), None).is_err());
        assert!(DoryVirtualKvMapper::from_model_metadata(meta, None, Some("invalid")).is_err());
    }
}
