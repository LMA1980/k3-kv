//! Wave 4 & O(1) recurrent shared-memory integration tests:
//! - 128-byte vector plane alignment (`VECTOR_PLANE_ALIGNMENT = 128`) across KV dtypes
//! - Quantized (`F16` 2x, `Q8_0` ~3.76x) KV packing, zero-allocation F32 unpack & `ScratchWorkspace`
//! - `LimitMEMLOCK` admission gating (`InsufficientLockableMemory`, exit code 11)
//! - O(1) SSM/recurrent state sizing independent of `n_ctx` + ping-pong `ArenaHandle` execution

use std::mem::size_of;

use jero_core::ipc::layout::ArenaHeader;
use jero_core::ipc::protocol::{KV_DTYPE_F16, KV_DTYPE_F32, KV_DTYPE_Q8_0, StatusCode};
use jero_core::ipc::shm::SharedMemoryArena;
use jero_core::{KvDtype, ScratchWorkspace};
use jero_kv_cache::{
    ArenaAllocationConfig, ArenaHandle, ArenaKvReader, KvError, KvHolder, LayerKvPolicy,
    VECTOR_PLANE_ALIGNMENT, compute_arena_layout,
};
use zerocopy::FromBytes;

fn deterministic_kv_batch(
    layer: usize,
    start_pos: usize,
    seq_len: usize,
    kv_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total = seq_len * kv_dim;
    let mut k = Vec::with_capacity(total);
    let mut v = Vec::with_capacity(total);
    for rel in 0..seq_len {
        let pos = start_pos + rel;
        for d in 0..kv_dim {
            let bucket = u8::try_from((layer * 29 + pos * 7 + d * 3) % 64).expect("< 64");
            let phase = f32::from(bucket) / 31.5 - 1.0;
            let pos_bias = f32::from(u8::try_from(pos % 5).expect("< 5")) * 0.05;
            let dim_bias = f32::from(u8::try_from(d % 7).expect("< 7")) * 0.04;
            k.push(phase * 0.85 + pos_bias);
            v.push(-phase * 0.75 + dim_bias);
        }
    }
    (k, v)
}

fn measure_drift(expected: &[f32], actual: &[f32]) -> (f32, f64) {
    assert_eq!(expected.len(), actual.len());
    let mut max_abs = 0.0_f32;
    let mut dot = 0.0_f64;
    let mut norm_exp = 0.0_f64;
    let mut norm_act = 0.0_f64;
    for (&e, &a) in expected.iter().zip(actual.iter()) {
        let diff = (e - a).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        dot += f64::from(e) * f64::from(a);
        norm_exp += f64::from(e) * f64::from(e);
        norm_act += f64::from(a) * f64::from(a);
    }
    let cosine = if norm_exp == 0.0 || norm_act == 0.0 {
        1.0
    } else {
        dot / (norm_exp.sqrt() * norm_act.sqrt())
    };
    (max_abs, cosine)
}

#[derive(Clone, Copy)]
struct RoundtripCase {
    dtype: u8,
    n_ctx: u32,
    head_dim: u32,
    head_count_kv: u32,
    num_layers: u32,
    max_abs_tol: f32,
    min_cos: f64,
}

fn run_quantized_roundtrip_case(
    holder: &KvHolder,
    scratch: &mut ScratchWorkspace,
    case: RoundtripCase,
) {
    let RoundtripCase {
        dtype,
        n_ctx,
        head_dim,
        head_count_kv,
        num_layers,
        max_abs_tol,
        min_cos,
    } = case;
    let cfg = ArenaAllocationConfig::new("wave4-fd-roundtrip", n_ctx, head_dim, num_layers)
        .with_head_count_kv(head_count_kv)
        .with_vocab_size(256)
        .with_kv_dtype(dtype);
    let layout = compute_arena_layout(&cfg).expect("layout");
    let align = VECTOR_PLANE_ALIGNMENT as u64;
    assert_eq!(layout.token_slots_offset % align, 0);
    assert_eq!(layout.kv_blocks_offset % align, 0);
    assert_eq!(layout.output_slabs_offset % align, 0);
    assert_eq!(layout.kv_plane_stride_bytes % align, 0);

    let alloc = holder.allocate(&cfg).expect("allocate wave4 arena");
    let len = usize::try_from(alloc.total_bytes).expect("usize");
    let writable_fd = holder.get_arena_fd(alloc.arena_id).expect("writable fd");
    let readonly_fd = holder.get_arena_fd(alloc.arena_id).expect("readonly fd");

    let mut rw =
        SharedMemoryArena::from_existing_fd(writable_fd, len, "wave4_rw", false).expect("rw map");
    let ro =
        SharedMemoryArena::from_existing_fd(readonly_fd, len, "wave4_ro", true).expect("ro map");

    let kv_dim = (head_count_kv * head_dim) as usize;
    let prefill_len = 36_usize;
    let decode_len = 8_usize;
    let total_len = prefill_len + decode_len;

    let mut handle = ArenaHandle::new(&mut rw).expect("ArenaHandle");
    {
        let mut writer = handle.kv_writer().expect("kv_writer");
        assert_eq!(writer.kv_dtype(), KvDtype::from_u8(dtype).unwrap());
        for layer in 0..(num_layers as usize) {
            let (k_pre, v_pre) = deterministic_kv_batch(layer, 0, prefill_len, kv_dim);
            assert_eq!(
                writer
                    .append_quantized_kv_batch(layer, 0, &k_pre, &v_pre)
                    .expect("append prefill"),
                prefill_len
            );
            let (k_dec, v_dec) = deterministic_kv_batch(layer, prefill_len, decode_len, kv_dim);
            assert_eq!(
                writer
                    .append_quantized_kv_batch(layer, prefill_len, &k_dec, &v_dec)
                    .expect("append decode"),
                total_len
            );
        }
    }
    handle
        .commit_prefill_range(0, prefill_len)
        .expect("commit prefill");
    handle
        .commit_token_span(prefill_len, decode_len)
        .expect("commit decode");
    assert_eq!(handle.committed_tokens().expect("committed"), total_len);

    let reader = ArenaKvReader::from_arena(&ro).expect("ro ArenaKvReader");
    assert_eq!(reader.committed_tokens().expect("ro committed"), total_len);
    assert_eq!(reader.kv_dtype(), KvDtype::from_u8(dtype).unwrap());

    let mut k_buf = vec![0.0_f32; total_len * kv_dim];
    let mut v_buf = vec![0.0_f32; total_len * kv_dim];
    for layer in 0..(num_layers as usize) {
        let (k_plane, v_plane) = reader.layer_kv_bytes(layer).expect("layer bytes");
        assert_eq!((k_plane.as_ptr() as usize) % VECTOR_PLANE_ALIGNMENT, 0);
        assert_eq!((v_plane.as_ptr() as usize) % VECTOR_PLANE_ALIGNMENT, 0);

        let (expected_k, expected_v) = deterministic_kv_batch(layer, 0, total_len, kv_dim);
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

        let (k_err, k_cos) = measure_drift(&expected_k, sk);
        let (v_err, v_cos) = measure_drift(&expected_v, sv);
        assert!(k_err <= max_abs_tol, "K max_abs {k_err} > {max_abs_tol}");
        assert!(v_err <= max_abs_tol, "V max_abs {v_err} > {max_abs_tol}");
        assert!(k_cos >= min_cos, "K cosine {k_cos} < {min_cos}");
        assert!(v_cos >= min_cos, "V cosine {v_cos} < {min_cos}");
    }

    holder.deallocate(alloc.arena_id).expect("deallocate");
}

#[test]
fn test_wave4_f16_2x_and_q8_0_376x_footprint_savings_and_memlock_admission() {
    // Full Llama-3-8B KV geometry layout verification (8K context): F32 = 4 GiB, F16 = 2 GiB, Q8_0 = 1.0625 GiB.
    let llama_f32 = ArenaAllocationConfig::new("llama3-8b", 8192, 128, 32)
        .with_head_count_kv(8)
        .with_vocab_size(128_256)
        .with_kv_dtype(KV_DTYPE_F32);
    let llama_f16 = llama_f32.clone().with_kv_dtype(KV_DTYPE_F16);
    let llama_q8 = llama_f32.clone().with_kv_dtype(KV_DTYPE_Q8_0);

    let l_f32 = compute_arena_layout(&llama_f32).expect("llama f32 layout");
    let l_f16 = compute_arena_layout(&llama_f16).expect("llama f16 layout");
    let l_q8 = compute_arena_layout(&llama_q8).expect("llama q8 layout");

    let kv_f32 = l_f32.output_slabs_offset - l_f32.kv_blocks_offset;
    let kv_f16 = l_f16.output_slabs_offset - l_f16.kv_blocks_offset;
    let kv_q8 = l_q8.output_slabs_offset - l_q8.kv_blocks_offset;

    assert_eq!(kv_f32, 32 * 2 * 8192 * 1024 * 4);
    assert_eq!(
        kv_f16,
        kv_f32 / 2,
        "F16 KV cache must be exactly 2.00x smaller than F32"
    );
    assert_eq!(
        kv_q8,
        32 * 2 * 8192 * (1024 / 32) * 34,
        "Q8_0 KV cache must use 34 bytes per 32-element group"
    );
    let kv_f32_kib = u32::try_from(kv_f32 / 1024).expect("KiB fits u32");
    let kv_q8_kib = u32::try_from(kv_q8 / 1024).expect("KiB fits u32");
    let q8_ratio = f64::from(kv_f32_kib) / f64::from(kv_q8_kib);
    assert!(
        (q8_ratio - (128.0 / 34.0)).abs() < 1e-6,
        "Q8_0 savings ratio should be ~3.7647x, got {q8_ratio}"
    );

    // Real SharedMemoryArena LimitMEMLOCK admission under constrained ceiling.
    let alloc_f32 = ArenaAllocationConfig::new("qwen-kv-ceiling", 256, 64, 4)
        .with_head_count_kv(4)
        .with_vocab_size(512)
        .with_kv_dtype(KV_DTYPE_F32);
    let alloc_f16 = alloc_f32.clone().with_kv_dtype(KV_DTYPE_F16);
    let alloc_q8 = alloc_f32.clone().with_kv_dtype(KV_DTYPE_Q8_0);

    let g_f32 = compute_arena_layout(&alloc_f32).expect("alloc f32");
    let g_f16 = compute_arena_layout(&alloc_f16).expect("alloc f16");
    let g_q8 = compute_arena_layout(&alloc_q8).expect("alloc q8");
    assert!(g_q8.total_bytes < g_f16.total_bytes && g_f16.total_bytes < g_f32.total_bytes);

    let holder_f16_cap = KvHolder::with_memlock_ceiling(g_f16.total_bytes + 4096);
    let f32_err = holder_f16_cap
        .allocate(&alloc_f32)
        .expect_err("F32 exceeds F16-sized LimitMEMLOCK ceiling");
    assert!(matches!(
        f32_err,
        KvError::InsufficientLockableMemory { .. }
    ));
    assert_eq!(
        f32_err.status_code(),
        StatusCode::InsufficientLockableMemory
    );
    assert_eq!(f32_err.exit_code(), 11);

    let res_f16 = holder_f16_cap
        .allocate(&alloc_f16)
        .expect("F16 fits inside LimitMEMLOCK ceiling");
    holder_f16_cap
        .deallocate(res_f16.arena_id)
        .expect("deallocate F16");

    let holder_q8_cap = KvHolder::with_memlock_ceiling(g_q8.total_bytes + 4096);
    assert_eq!(
        holder_q8_cap
            .allocate(&alloc_f16)
            .expect_err("F16 exceeds Q8_0-sized LimitMEMLOCK ceiling")
            .exit_code(),
        11
    );
    let res_q8 = holder_q8_cap
        .allocate(&alloc_q8)
        .expect("Q8_0 fits inside tighter LimitMEMLOCK ceiling");
    holder_q8_cap
        .deallocate(res_q8.arena_id)
        .expect("deallocate Q8_0");
}

#[test]
fn test_wave4_shared_memory_arena_128b_plane_alignment_and_quantized_prefill_decode_roundtrip() {
    let holder = KvHolder::new();
    let cases = [
        RoundtripCase {
            dtype: KV_DTYPE_F32,
            n_ctx: 64,
            head_dim: 64,
            head_count_kv: 2,
            num_layers: 4,
            max_abs_tol: 0.0,
            min_cos: 0.999_999_9,
        },
        RoundtripCase {
            dtype: KV_DTYPE_F16,
            n_ctx: 64,
            head_dim: 64,
            head_count_kv: 2,
            num_layers: 4,
            max_abs_tol: 1e-3,
            min_cos: 0.999_99,
        },
        RoundtripCase {
            dtype: KV_DTYPE_Q8_0,
            n_ctx: 64,
            head_dim: 64,
            head_count_kv: 2,
            num_layers: 4,
            max_abs_tol: 1.5e-2,
            min_cos: 0.999,
        },
        // Non-128B raw plane Q8_0 case: 48 tok * 34 B = 1632 B -> aligned stride 1664 B (13 * 128)
        RoundtripCase {
            dtype: KV_DTYPE_Q8_0,
            n_ctx: 48,
            head_dim: 32,
            head_count_kv: 1,
            num_layers: 4,
            max_abs_tol: 1.5e-2,
            min_cos: 0.999,
        },
    ];

    let mut scratch = ScratchWorkspace::new(256 * 1024).expect("scratch workspace");
    for case in cases {
        run_quantized_roundtrip_case(&holder, &mut scratch, case);
    }
}

#[test]
fn test_wave4_sliding_window_and_subrange_quantized_reads() {
    let holder = KvHolder::new();
    let cfg = ArenaAllocationConfig::new("wave4-swa", 32, 32, 2)
        .with_head_count_kv(1)
        .with_vocab_size(64)
        .with_kv_dtype(KV_DTYPE_Q8_0)
        .with_layer_policies(vec![
            LayerKvPolicy::SlidingWindow { window_tokens: 16 },
            LayerKvPolicy::GlobalFullContext,
        ]);
    let layout = compute_arena_layout(&cfg).expect("hybrid layout");
    assert_eq!(layout.k_plane_offset(0) % VECTOR_PLANE_ALIGNMENT as u64, 0);
    assert_eq!(layout.v_plane_offset(0) % VECTOR_PLANE_ALIGNMENT as u64, 0);
    assert_eq!(layout.k_plane_offset(1) % VECTOR_PLANE_ALIGNMENT as u64, 0);
    assert_eq!(layout.v_plane_offset(1) % VECTOR_PLANE_ALIGNMENT as u64, 0);

    // Uniform Q8_0 slab with header SWA window active at position 20 -> active window [8..20).
    let uniform_cfg = ArenaAllocationConfig::new("wave4-swa-read", 32, 32, 2)
        .with_head_count_kv(1)
        .with_vocab_size(64)
        .with_kv_dtype(KV_DTYPE_Q8_0);
    let alloc = holder.allocate(&uniform_cfg).expect("alloc");
    let len = usize::try_from(alloc.total_bytes).expect("usize");
    let fd = holder.get_arena_fd(alloc.arena_id).expect("fd");
    let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "wave4_swa", false).expect("rw");
    {
        let (hdr_mut, _) = ArenaHeader::mut_from_prefix(rw.as_mut_slice()).expect("hdr");
        hdr_mut.swa_window_size = 12;
        assert_eq!(size_of::<ArenaHeader>() % 64, 0);
    }

    let mut handle = ArenaHandle::new(&mut rw).expect("handle");
    {
        let mut writer = handle.kv_writer().expect("writer");
        let (k, v) = deterministic_kv_batch(0, 0, 20, 32);
        writer
            .append_quantized_kv_batch(0, 0, &k, &v)
            .expect("append 20");
    }
    handle.commit_prefill_range(0, 20).expect("commit 20");

    let mut scratch = ScratchWorkspace::new(64 * 1024).expect("scratch");
    let reader = handle.kv_reader().expect("reader");
    let scope = scratch.scope();
    let (sk, sv) = reader
        .read_committed_kv_range_in_scratch(0, &scope)
        .expect("SWA committed scratch read");
    assert_eq!(sk.len(), 12 * 32);
    assert_eq!(sv.len(), 12 * 32);

    let (expected_k, expected_v) = deterministic_kv_batch(0, 8, 12, 32);
    let (k_err, k_cos) = measure_drift(&expected_k, sk);
    let (v_err, v_cos) = measure_drift(&expected_v, sv);
    assert!(k_err <= 1.5e-2 && k_cos >= 0.999);
    assert!(v_err <= 1.5e-2 && v_cos >= 0.999);

    holder.deallocate(alloc.arena_id).expect("dealloc");
}

#[test]
fn test_recurrent_o1_memlock_admission_and_ping_pong_across_context_scales() {
    let base_64 = ArenaAllocationConfig::new("qwen3.5-ssm", 64, 64, 0)
        .with_vocab_size(512)
        .with_recurrent_state(24, 4096, 32_768);
    let ctx_mid = ArenaAllocationConfig::new("qwen3.5-ssm", 4096, 64, 0)
        .with_vocab_size(512)
        .with_recurrent_state(24, 4096, 32_768);
    let ctx_128k = ArenaAllocationConfig::new("qwen3.5-ssm", 131_072, 64, 0)
        .with_vocab_size(512)
        .with_recurrent_state(24, 4096, 32_768);

    let layout_64 = compute_arena_layout(&base_64).expect("layout 64");
    let layout_mid = compute_arena_layout(&ctx_mid).expect("layout mid");
    let layout_128k = compute_arena_layout(&ctx_128k).expect("layout 128k");

    // O(1) footprint identity independent of sequence context length.
    assert_eq!(layout_64.total_bytes, layout_mid.total_bytes);
    assert_eq!(layout_64.total_bytes, layout_128k.total_bytes);
    assert_eq!(
        layout_64.recurrent_total_bytes,
        layout_128k.recurrent_total_bytes
    );
    assert_eq!(
        base_64.estimate_recurrent_state_bytes().expect("est"),
        layout_128k.recurrent_total_bytes
    );

    // Bounded LimitMEMLOCK admission succeeds at 131_072 tokens with fixed O(1) budget.
    let holder = KvHolder::with_memlock_ceiling(layout_64.total_bytes + 4096);
    let alloc = holder
        .allocate(&ctx_128k)
        .expect("allocate 128k O(1) arena");
    assert_eq!(alloc.total_bytes, layout_64.total_bytes);

    let fd = holder.get_arena_fd(alloc.arena_id).expect("fd");
    let len = usize::try_from(alloc.total_bytes).expect("usize");
    let mut rw = SharedMemoryArena::from_existing_fd(fd, len, "ssm_128k", false).expect("rw");
    let mut handle = ArenaHandle::new(&mut rw).expect("ArenaHandle");

    let hdr0 = handle.recurrent_header().expect("recurrent_header");
    assert_eq!(hdr0.num_recurrent_layers, 24);
    assert_eq!(hdr0.committed_step, 0);

    {
        let (_hdr, active, inactive) = handle.recurrent_banks_mut().expect("banks");
        assert!(active.iter().all(|&b| b == 0));
        inactive.fill(0x3C);
    }
    let hdr1 = handle.commit_recurrent_step(1).expect("commit step 1");
    assert_eq!(hdr1.committed_step, 1);

    // Simulate torn crash write to inactive bank without committing; active checkpoint stays intact.
    {
        let (_hdr, active, inactive) = handle.recurrent_banks_mut().expect("post-commit banks");
        assert!(active.iter().all(|&b| b == 0x3C));
        inactive.fill(0xDE);
    }
    let hdr_check = handle
        .recurrent_header()
        .expect("header after uncommitted write");
    assert_eq!(hdr_check.committed_step, 1);
    let (_hdr, active_after, _) = handle.recurrent_banks_mut().expect("banks recovery");
    assert!(active_after.iter().all(|&b| b == 0x3C));

    holder.deallocate(alloc.arena_id).expect("deallocate");
}
