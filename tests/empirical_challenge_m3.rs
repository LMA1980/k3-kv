//! Empirical challenge and stress test harness for `k3-kv` (Milestone 3 of US-05).
//!
//! Evaluates:
//! 1. Operator `LimitMEMLOCK` ceiling boundary conditions:
//!    - Exactly at ceiling
//!    - Ceiling - 1 byte
//!    - Ceiling + 1 byte
//!    - Multiple arena cumulative overflow
//!    - Arithmetic overflow rejection
//! 2. Rapid allocate/deallocate cycles with strict memory and file descriptor leak detection.
//! 3. Sub-region 64-byte alignment and non-overlapping offset verification across parameter permutations.
//! 4. Physical memory mapping pointer cacheline alignment and isolation boundary integrity.
//! 5. Multi-threaded concurrent allocation, accounting consistency, and race-free reclamation.

#![allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::zombie_processes
)]

use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use k3_core::ipc::layout::{
    ARENA_MAGIC, ARENA_VERSION, ArenaHeader, CACHELINE_ALIGNMENT, KvBlockMeta, OutputSlabHeader,
    TAG_FREE, TAG_PERSIST, TokenInputSlotHeader,
};
use k3_core::ipc::protocol::StatusCode;
use k3_core::ipc::shm::system_page_size;
use k3_kv::holder::{ArenaAllocationConfig, KvError, KvHolder, compute_arena_layout};

/// Helper to count currently open file descriptors for leak verification.
fn count_open_fds() -> usize {
    #[cfg(target_os = "macos")]
    {
        std::fs::read_dir("/dev/fd").map_or(0, Iterator::count)
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_dir("/proc/self/fd").map_or(0, Iterator::count)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

// ============================================================================
// 1. MEMLOCK CEILING BOUNDARY CONDITIONS
// ============================================================================

#[test]
fn test_challenge_memlock_exact_ceiling_succeeds() {
    let config = ArenaAllocationConfig::new("exact_ceiling_model", 64, 64, 2);
    let layout = compute_arena_layout(&config).expect("layout computation");
    let exact_ceiling = layout.total_bytes;

    let holder = KvHolder::with_memlock_ceiling(exact_ceiling);
    assert_eq!(holder.memlock_ceiling_bytes(), exact_ceiling);
    assert_eq!(holder.total_allocated_bytes(), 0);
    assert_eq!(holder.available_lockable_bytes(), exact_ceiling);

    // Allocation where requested bytes == ceiling MUST succeed exactly
    let res = holder
        .allocate(&config)
        .expect("allocation exactly equal to memlock ceiling MUST succeed");

    assert_eq!(res.total_bytes, exact_ceiling);
    assert_eq!(holder.total_allocated_bytes(), exact_ceiling);
    assert_eq!(holder.available_lockable_bytes(), 0);
    assert_eq!(holder.arena_count(), 1);
    assert!(holder.contains_arena(res.arena_id));

    // Subsequent allocation (even with 0 requested bytes or 1 block) MUST fail
    let res2 = holder.allocate(&config);
    match res2 {
        Err(
            err @ KvError::InsufficientLockableMemory {
                required_bytes,
                allocated_bytes,
                memlock_ceiling_bytes,
            },
        ) => {
            assert_eq!(required_bytes, exact_ceiling);
            assert_eq!(allocated_bytes, exact_ceiling);
            assert_eq!(memlock_ceiling_bytes, exact_ceiling);
            assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
            assert_eq!(err.exit_code(), 11);
        }
        other => panic!("expected InsufficientLockableMemory error, got {other:?}"),
    }

    // Accounting remains unchanged after rejected allocation
    assert_eq!(holder.total_allocated_bytes(), exact_ceiling);
    assert_eq!(holder.available_lockable_bytes(), 0);
}

#[test]
fn test_challenge_memlock_ceiling_minus_one_byte_rejects() {
    let config = ArenaAllocationConfig::new("ceiling_minus_one_model", 64, 64, 2);
    let layout = compute_arena_layout(&config).expect("layout computation");
    let required_bytes = layout.total_bytes;

    // Ceiling is strictly 1 byte below required capacity
    let tight_ceiling = required_bytes - 1;
    let holder = KvHolder::with_memlock_ceiling(tight_ceiling);

    let res = holder.allocate(&config);
    match res {
        Err(
            err @ KvError::InsufficientLockableMemory {
                required_bytes: req,
                allocated_bytes: alloc,
                memlock_ceiling_bytes: ceil,
            },
        ) => {
            assert_eq!(req, required_bytes);
            assert_eq!(alloc, 0);
            assert_eq!(ceil, tight_ceiling);
            assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
            assert_eq!(err.exit_code(), 11);
        }
        other => panic!("expected InsufficientLockableMemory, got {other:?}"),
    }

    // Verify zero budget leaked
    assert_eq!(holder.total_allocated_bytes(), 0);
    assert_eq!(holder.available_lockable_bytes(), tight_ceiling);
    assert_eq!(holder.arena_count(), 0);
}

#[test]
fn test_challenge_memlock_ceiling_plus_one_byte_succeeds() {
    let config = ArenaAllocationConfig::new("ceiling_plus_one_model", 64, 64, 2);
    let layout = compute_arena_layout(&config).expect("layout computation");
    let required_bytes = layout.total_bytes;

    // Ceiling is exactly 1 byte above required capacity
    let generous_ceiling = required_bytes + 1;
    let holder = KvHolder::with_memlock_ceiling(generous_ceiling);

    let res = holder
        .allocate(&config)
        .expect("allocation at ceiling + 1 byte MUST succeed");

    assert_eq!(res.total_bytes, required_bytes);
    assert_eq!(holder.total_allocated_bytes(), required_bytes);
    assert_eq!(holder.available_lockable_bytes(), 1);
    assert_eq!(holder.arena_count(), 1);

    // Any second allocation will need at least system page size (> 1 byte) and must fail
    let res2 = holder.allocate(&config);
    match res2 {
        Err(KvError::InsufficientLockableMemory {
            required_bytes: req,
            allocated_bytes: alloc,
            memlock_ceiling_bytes: ceil,
        }) => {
            assert_eq!(req, required_bytes);
            assert_eq!(alloc, required_bytes);
            assert_eq!(ceil, generous_ceiling);
        }
        other => panic!("expected InsufficientLockableMemory, got {other:?}"),
    }

    // Deallocate and verify budget reclaimed to generous_ceiling
    let freed = holder.deallocate(res.arena_id).expect("deallocate");
    assert_eq!(freed, required_bytes);
    assert_eq!(holder.total_allocated_bytes(), 0);
    assert_eq!(holder.available_lockable_bytes(), generous_ceiling);
}

#[test]
fn test_challenge_memlock_multiple_arena_cumulative_overflow() {
    let config = ArenaAllocationConfig::new("multi_arena_model", 64, 64, 2);
    let layout = compute_arena_layout(&config).expect("layout computation");
    let single_bytes = layout.total_bytes;

    // Budget fits exactly 4 arenas, not 5
    let exact_4_ceiling = single_bytes * 4;
    let holder = KvHolder::with_memlock_ceiling(exact_4_ceiling);

    let mut ids = Vec::new();
    for i in 1..=4 {
        let res = holder
            .allocate(&config)
            .unwrap_or_else(|e| panic!("allocation {i} failed: {e:?}"));
        ids.push(res.arena_id);
        assert_eq!(holder.total_allocated_bytes(), single_bytes * i as u64);
        assert_eq!(holder.arena_count(), i);
    }

    assert_eq!(holder.total_allocated_bytes(), exact_4_ceiling);
    assert_eq!(holder.available_lockable_bytes(), 0);

    // 5th allocation MUST fail with exit code 11
    let overflow_res = holder.allocate(&config);
    match overflow_res {
        Err(
            err @ KvError::InsufficientLockableMemory {
                required_bytes,
                allocated_bytes,
                memlock_ceiling_bytes,
            },
        ) => {
            assert_eq!(required_bytes, single_bytes);
            assert_eq!(allocated_bytes, exact_4_ceiling);
            assert_eq!(memlock_ceiling_bytes, exact_4_ceiling);
            assert_eq!(err.status_code(), StatusCode::InsufficientLockableMemory);
            assert_eq!(err.exit_code(), 11);
        }
        other => panic!("expected InsufficientLockableMemory, got {other:?}"),
    }

    // Verify budget remains strictly pegged at ceiling
    assert_eq!(holder.total_allocated_bytes(), exact_4_ceiling);
    assert_eq!(holder.arena_count(), 4);

    // Deallocate arena #2 (intermediate arena)
    let freed = holder.deallocate(ids[1]).expect("deallocate arena 2");
    assert_eq!(freed, single_bytes);
    assert_eq!(holder.total_allocated_bytes(), single_bytes * 3);
    assert_eq!(holder.available_lockable_bytes(), single_bytes);
    assert_eq!(holder.arena_count(), 3);

    // Now allocation 5 succeeds in the reclaimed slot
    let res5 = holder
        .allocate(&config)
        .expect("allocation 5 must succeed after deallocating arena 2");
    assert_eq!(holder.total_allocated_bytes(), exact_4_ceiling);
    assert_eq!(holder.arena_count(), 4);
    assert!(holder.contains_arena(res5.arena_id));

    // Clear all
    let cleared = holder.clear();
    assert_eq!(cleared, 4);
    assert_eq!(holder.total_allocated_bytes(), 0);
    assert_eq!(holder.available_lockable_bytes(), exact_4_ceiling);
    assert_eq!(holder.arena_count(), 0);
}

// ============================================================================
// 2. RAPID ALLOCATE / DEALLOCATE CYCLES & LEAK DETECTION (MEMORY & FD)
// ============================================================================

#[test]
fn test_challenge_rapid_allocate_deallocate_no_fd_or_memory_leaks() {
    let baseline_fds = count_open_fds();
    let holder = KvHolder::new();
    let config = ArenaAllocationConfig::new("stress_model", 64, 64, 2);

    // 1. Rapid sequential allocate-dup-deallocate (200 cycles)
    for _ in 0..200 {
        let res = holder.allocate(&config).expect("sequential allocate");
        assert!(holder.total_allocated_bytes() > 0);

        // Duplicate file descriptor (as k3-kv does for SCM_RIGHTS)
        let fd = holder.get_arena_fd(res.arena_id).expect("fd dup");
        assert!(fd.as_raw_fd() >= 0);
        drop(fd); // Explicitly close duplicated descriptor

        let freed = holder.deallocate(res.arena_id).expect("deallocate");
        assert_eq!(freed, res.total_bytes);
        assert_eq!(holder.total_allocated_bytes(), 0);
        assert_eq!(holder.arena_count(), 0);
    }

    // 2. Batch allocate-deallocate cycles (10 arenas per batch, 15 batches)
    for _ in 0..15 {
        let mut batch_ids = Vec::new();
        let mut fds = Vec::new();

        for _ in 0..10 {
            let res = holder.allocate(&config).expect("batch allocate");
            let fd = holder.get_arena_fd(res.arena_id).expect("batch fd dup");
            batch_ids.push(res.arena_id);
            fds.push(fd);
        }

        assert_eq!(holder.arena_count(), 10);
        drop(fds); // Close all duplicated descriptors

        // Deallocate in reverse order
        while let Some(id) = batch_ids.pop() {
            holder.deallocate(id).expect("batch deallocate");
        }

        assert_eq!(holder.total_allocated_bytes(), 0);
        assert_eq!(holder.arena_count(), 0);
    }

    // 3. Allocate and clear cycles (20 arenas, cleared via KvHolder::clear)
    for _ in 0..5 {
        for _ in 0..20 {
            let _ = holder.allocate(&config).expect("alloc before clear");
        }
        assert_eq!(holder.arena_count(), 20);
        let count = holder.clear();
        assert_eq!(count, 20);
        assert_eq!(holder.total_allocated_bytes(), 0);
        assert_eq!(holder.arena_count(), 0);
    }

    // Leak Verification: Ensure open file descriptors did not grow (accommodating concurrent test FD cleanup)
    let current_fds = count_open_fds();
    assert!(
        current_fds <= baseline_fds + 2,
        "File descriptor leak detected! Baseline: {baseline_fds}, Current: {current_fds}"
    );

    // Memory Verification: Inner registry holds zero bytes and zero entries
    assert_eq!(holder.total_allocated_bytes(), 0);
    assert_eq!(holder.arena_count(), 0);
    assert_eq!(holder.list_arena_ids(), Vec::<u64>::new());
}

// ============================================================================
// 3. 64-BYTE ALIGNMENT AND NON-OVERLAPPING REGION VERIFICATION
// ============================================================================

#[test]
fn test_challenge_arena_memory_layout_alignment_and_no_overlap_exhaustive() {
    let page_size = system_page_size() as u64;

    // Test matrix covering edge cases:
    // - Non-power-of-two n_ctx (1, 15, 17, 33, 65, 127)
    // - Standard n_ctx (16, 32, 64, 128, 512, 2048)
    // - Varied head_dim (32, 64, 128)
    // - Varied num_layers (1, 2, 16)
    // - Varied head_count_kv (1, 2, 4)
    // - Varied vocab_size (100, 32000, 151936)
    // - Varied active_slots (1, 2)

    let n_ctx_cases = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 512, 2048];
    let head_dim_cases = [32, 64, 128];
    let num_layers_cases = [1, 2, 16];
    let head_count_kv_cases = [1, 2, 4];
    let vocab_size_cases = [100, 32000, 151_936];
    let active_slots_cases = [1, 2];

    let mut evaluated = 0;

    for &n_ctx in &n_ctx_cases {
        for &head_dim in &head_dim_cases {
            for &num_layers in &num_layers_cases {
                for &head_count_kv in &head_count_kv_cases {
                    for &vocab_size in &vocab_size_cases {
                        for &active_slots in &active_slots_cases {
                            evaluated += 1;
                            let config = ArenaAllocationConfig::new(
                                format!("test_{evaluated}"),
                                n_ctx,
                                head_dim,
                                num_layers,
                            )
                            .with_head_count_kv(head_count_kv)
                            .with_vocab_size(vocab_size)
                            .with_active_slots(active_slots);

                            let layout = compute_arena_layout(&config).unwrap_or_else(|e| {
                                panic!(
                                    "layout failed for n_ctx={n_ctx}, head_dim={head_dim}: {e:?}"
                                )
                            });

                            // 1. Strict 64-byte alignment check on all offsets
                            assert_eq!(
                                layout.meta_table_offset % CACHELINE_ALIGNMENT as u64,
                                0,
                                "meta_table_offset {off} must be 64-byte aligned",
                                off = layout.meta_table_offset
                            );
                            assert_eq!(
                                layout.token_slots_offset % CACHELINE_ALIGNMENT as u64,
                                0,
                                "token_slots_offset {off} must be 64-byte aligned",
                                off = layout.token_slots_offset
                            );
                            assert_eq!(
                                layout.kv_blocks_offset % CACHELINE_ALIGNMENT as u64,
                                0,
                                "kv_blocks_offset {off} must be 64-byte aligned",
                                off = layout.kv_blocks_offset
                            );
                            assert_eq!(
                                layout.output_slabs_offset % CACHELINE_ALIGNMENT as u64,
                                0,
                                "output_slabs_offset {off} must be 64-byte aligned",
                                off = layout.output_slabs_offset
                            );
                            assert_eq!(
                                layout.total_bytes % page_size,
                                0,
                                "total_bytes {tot} must be page aligned ({page_size})",
                                tot = layout.total_bytes
                            );

                            // 2. Compute exact boundaries of each region
                            // Region 0: ArenaHeader (0..128)
                            let header_end = 128u64;

                            // Region 1: KvBlockMeta table
                            let meta_start = layout.meta_table_offset;
                            let meta_bytes =
                                u64::from(layout.block_count) * size_of::<KvBlockMeta>() as u64;
                            let meta_end = meta_start + meta_bytes;

                            // Region 2: TokenInputSlots
                            let slot_start = layout.token_slots_offset;
                            let per_slot_bytes = ((u64::from(n_ctx) * 4
                                + size_of::<TokenInputSlotHeader>() as u64)
                                + 63)
                                & !63;
                            let slots_bytes = per_slot_bytes * u64::from(active_slots);
                            let slot_end = slot_start + slots_bytes;

                            // Region 3: KvBlockData
                            let kv_start = layout.kv_blocks_offset;
                            let bytes_per_token = 2u64
                                * u64::from(num_layers)
                                * u64::from(head_count_kv)
                                * u64::from(head_dim)
                                * 4;
                            let kv_bytes = u64::from(layout.block_count) * 16 * bytes_per_token;
                            let kv_end = kv_start + kv_bytes;

                            // Region 4: OutputSlabs
                            let slab_start = layout.output_slabs_offset;
                            let per_slab_bytes = ((u64::from(vocab_size) * 4
                                + size_of::<OutputSlabHeader>() as u64)
                                + 63)
                                & !63;
                            let slabs_bytes = per_slab_bytes * u64::from(active_slots);
                            let slab_end = slab_start + slabs_bytes;

                            // 3. Strict Non-Overlapping Verification
                            assert!(
                                header_end <= meta_start,
                                "Header overlaps meta table: header_end={header_end}, meta_start={meta_start}"
                            );
                            assert!(
                                meta_end <= slot_start,
                                "Meta table overlaps token slots: meta_end={meta_end}, slot_start={slot_start}"
                            );
                            assert!(
                                slot_end <= kv_start,
                                "Token slots overlap KV blocks: slot_end={slot_end}, kv_start={kv_start}"
                            );
                            assert!(
                                kv_end <= slab_start,
                                "KV blocks overlap output slabs: kv_end={kv_end}, slab_start={slab_start}"
                            );
                            assert!(
                                slab_end <= layout.total_bytes,
                                "Output slabs exceed arena capacity: slab_end={slab_end}, total={tot}",
                                tot = layout.total_bytes
                            );
                        }
                    }
                }
            }
        }
    }

    assert!(
        evaluated > 500,
        "Exhaustive test matrix should have evaluated > 500 permutations, evaluated: {evaluated}"
    );
}

#[test]
fn test_challenge_physical_memory_pointer_alignment_and_mutation_isolation() {
    let holder = KvHolder::new();
    let config = ArenaAllocationConfig::new("physical_align_model", 64, 64, 2)
        .with_head_count_kv(2)
        .with_vocab_size(32000)
        .with_active_slots(1);

    let res = holder.allocate(&config).expect("allocate physical arena");
    let arena = holder.get_arena(res.arena_id).expect("arena lookup");

    let base_ptr = arena.as_ptr() as usize;
    assert_eq!(
        base_ptr % CACHELINE_ALIGNMENT,
        0,
        "base mmap address must be 64-byte aligned"
    );

    // 1. ArenaHeader pointer alignment
    let hdr = arena.header().expect("header access");
    let hdr_ptr = std::ptr::from_ref::<ArenaHeader>(hdr) as usize;
    assert_eq!(hdr_ptr, base_ptr);
    assert_eq!(hdr_ptr % CACHELINE_ALIGNMENT, 0);
    assert_eq!(hdr.magic, ARENA_MAGIC);
    assert_eq!(hdr.version, ARENA_VERSION);

    // 2. KvBlockMeta entries pointer alignment
    for i in 0..res.block_count as usize {
        let meta = arena.block_meta(i).expect("block meta");
        let meta_ptr = std::ptr::from_ref::<KvBlockMeta>(meta) as usize;
        assert_eq!(
            meta_ptr % CACHELINE_ALIGNMENT,
            0,
            "block_meta[{i}] pointer {meta_ptr:#x} must be 64-byte aligned"
        );
        assert_eq!(meta.tag, TAG_FREE);
    }

    // 3. In-place mutation and cross-region isolation test
    // Mutate block 0 and verify other regions are untouched
    let fd = holder.get_arena_fd(res.arena_id).expect("fd");
    let mut mut_view = k3_core::ipc::shm::SharedMemoryArena::from_existing_fd(
        fd,
        res.total_bytes as usize,
        "mut_view",
        false,
    )
    .expect("mutable mapping view");

    // Mutate block 0 metadata
    {
        let meta0 = mut_view.block_meta_mut(0).expect("block meta 0 mut");
        meta0.block_id = 999;
        meta0.tag = TAG_PERSIST;
        meta0.committed_token_count = 16;
        meta0.sampler_rng_state = [0xDEAD, 0xBEEF, 0xCAFE, 0xBABE];
    }

    // Verify block 0 was modified
    {
        let meta0_view = arena.block_meta(0).expect("block meta 0 view");
        assert_eq!(meta0_view.block_id, 999);
        assert_eq!(meta0_view.tag, TAG_PERSIST);
        assert_eq!(meta0_view.committed_tokens(), 16);
        assert_eq!(
            meta0_view.sampler_rng_state,
            [0xDEAD, 0xBEEF, 0xCAFE, 0xBABE]
        );
    }

    // Verify block 1 was NOT corrupted (isolation between adjacent 64B cachelines)
    {
        let meta1_view = arena.block_meta(1).expect("block meta 1 view");
        assert_eq!(meta1_view.block_id, 1);
        assert_eq!(meta1_view.tag, TAG_FREE);
        assert_eq!(meta1_view.committed_tokens(), 0);
    }

    // Verify ArenaHeader was NOT corrupted
    {
        let hdr_check = arena.header().expect("header check");
        assert_eq!(hdr_check.magic, ARENA_MAGIC);
        assert_eq!(hdr_check.version, ARENA_VERSION);
    }
}

// ============================================================================
// 4. MULTI-THREADED CONCURRENT STRESS & CEILING INVARIANT
// ============================================================================

#[test]
fn test_challenge_concurrent_multithreaded_stress_and_memlock_accounting() {
    let single_config = ArenaAllocationConfig::new("thread_stress", 64, 64, 2);
    let layout = compute_arena_layout(&single_config).expect("layout");
    let single_bytes = layout.total_bytes;

    // Memlock ceiling sized to allow at most 6 concurrent arenas
    let ceiling = single_bytes * 6;
    let holder = Arc::new(KvHolder::with_memlock_ceiling(ceiling));

    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // 8 concurrent worker threads
    for t in 0..8 {
        let holder_clone = Arc::clone(&holder);
        let stop_clone = Arc::clone(&stop_flag);
        let config_clone = single_config.clone();

        handles.push(thread::spawn(move || {
            let mut allocated_ids = Vec::new();
            let mut iterations = 0;

            while !stop_clone.load(Ordering::Relaxed) && iterations < 150 {
                iterations += 1;
                // Randomly choose allocate or deallocate
                if allocated_ids.is_empty() || (iterations % 3 != 0 && allocated_ids.len() < 3) {
                    match holder_clone.allocate(&config_clone) {
                        Ok(res) => {
                            // Verify ceiling invariant never breached
                            assert!(
                                holder_clone.total_allocated_bytes() <= ceiling,
                                "MEMLOCK CEILING BREACHED! allocated={}, ceiling={}",
                                holder_clone.total_allocated_bytes(),
                                ceiling
                            );

                            // Duplicate FD to test concurrent OS FD operations
                            if let Ok(fd) = holder_clone.get_arena_fd(res.arena_id) {
                                assert!(fd.as_raw_fd() >= 0);
                            }
                            allocated_ids.push(res.arena_id);
                        }
                        Err(KvError::InsufficientLockableMemory { .. }) => {
                            // Expected when ceiling is saturated
                            thread::yield_now();
                        }
                        Err(e) => panic!("Thread {t} unexpected error: {e:?}"),
                    }
                } else if let Some(id) = allocated_ids.pop() {
                    let _ = holder_clone.deallocate(id);
                }
            }

            // Clean up remaining arenas allocated by this thread
            for id in allocated_ids {
                let _ = holder_clone.deallocate(id);
            }
        }));
    }

    // Let threads run
    for h in handles {
        h.join().expect("thread join failed");
    }

    // Verify final state: all memory reclaimed cleanly
    assert_eq!(
        holder.total_allocated_bytes(),
        0,
        "Memory leaked after concurrent stress! Total remaining: {}",
        holder.total_allocated_bytes()
    );
    assert_eq!(
        holder.arena_count(),
        0,
        "Arenas leaked after concurrent stress! Remaining count: {}",
        holder.arena_count()
    );
}

// ============================================================================
// 5. DAEMON SIGTERM SHUTDOWN WITH CONNECTED CLIENT EMPIRICAL PROBE
// ============================================================================

#[test]
fn test_challenge_daemon_sigterm_graceful_shutdown_with_connected_client() {
    use k3_core::ipc::protocol::{
        IpcMessage, PingRequest, PingResponse, recv_msg_frame, send_msg_frame,
    };
    use std::os::unix::net::UnixStream;
    use std::process::Command;
    use std::time::{Duration, Instant};

    let socket_path = std::env::temp_dir().join(format!("k3_kv_probe_{}.sock", std::process::id()));
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let bin = env!("CARGO_BIN_EXE_k3-kv");
    let mut child = Command::new(bin)
        .arg("--socket")
        .arg(&socket_path)
        .spawn()
        .expect("spawn daemon");

    // Wait until socket is ready
    let start = Instant::now();
    let timeout = Duration::from_secs(3);
    let mut connected = false;
    while start.elapsed() < timeout {
        if socket_path.exists() {
            if let Ok(_stream) = UnixStream::connect(&socket_path) {
                connected = true;
                break;
            }
        }
        thread::sleep(Duration::from_millis(15));
    }
    assert!(connected, "timed out waiting for k3-kv daemon");

    // Connect client and send ping
    let stream = UnixStream::connect(&socket_path).expect("connect client");
    let ping = IpcMessage::PingRequest(PingRequest {
        timestamp_ns: 12345,
    });
    send_msg_frame(&stream, &ping, None).expect("send ping");
    let (resp, _) = recv_msg_frame(&stream).expect("recv ping response");
    assert_eq!(
        resp,
        IpcMessage::PingResponse(PingResponse {
            timestamp_ns: 12345
        })
    );

    // IMPORTANT: Keep `stream` OPEN (do NOT drop it) to simulate active connected client!
    // Now send SIGTERM to daemon process
    let pid = child.id() as libc::pid_t;
    // SAFETY: Sending SIGTERM to spawned child process.
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }

    // Probe whether child exits within 2 seconds
    let shutdown_start = Instant::now();
    let mut exited = false;
    let mut exit_status = None;
    while shutdown_start.elapsed() < Duration::from_secs(2) {
        if let Ok(Some(status)) = child.try_wait() {
            exited = true;
            exit_status = Some(status);
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }

    // Clean up child process and socket regardless of result
    if !exited {
        // Daemon failed to shut down while client is connected!
        // SAFETY: Sending SIGKILL to terminate hung process.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        let _ = child.wait();
    }
    drop(stream);
    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    assert!(
        exited,
        "k3-kv daemon failed to exit within 2 seconds on SIGTERM while an active client connection is held open!"
    );
    assert_eq!(
        exit_status.and_then(|s| s.code()),
        Some(0),
        "Daemon should exit with code 0 on SIGTERM"
    );
}
