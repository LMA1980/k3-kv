//! End-to-end integration tests for `k3-kv` daemon lifecycle, CLI, Fory IPC, and `SCM_RIGHTS` FD transfer.

use std::fs;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::sleep;
use std::time::{Duration, Instant};

use k3_core::ipc::layout::{ARENA_MAGIC, ARENA_VERSION, TAG_PERSIST};
use k3_core::ipc::protocol::{
    AllocateArenaRequest, DeallocateArenaRequest, GetArenaFdRequest, IpcMessage, PingRequest,
    PingResponse, StatusCode, recv_msg_frame, send_msg_frame,
};
use k3_core::ipc::shm::SharedMemoryArena;

static TEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Helper to generate a unique temporary socket path within OS limits (104 chars on macOS).
fn generate_test_socket_path() -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("k3_kv_test_{}_{}.sock", std::process::id(), id))
}

/// Helper struct that spawns and supervises a test daemon process with automatic cleanup.
struct DaemonHarness {
    socket_path: PathBuf,
    process: Option<Child>,
}

impl DaemonHarness {
    fn start(socket_path: PathBuf, max_memory_bytes: Option<u64>) -> Self {
        let bin = env!("CARGO_BIN_EXE_k3-kv");
        let mut cmd = Command::new(bin);
        cmd.arg("--socket").arg(&socket_path).arg("--verbose");

        if let Some(max_bytes) = max_memory_bytes {
            cmd.arg("--max-memory-bytes").arg(max_bytes.to_string());
        }

        let child = cmd.spawn().expect("failed to spawn k3-kv daemon");

        // Wait until socket file appears and accepts connections (up to 3 seconds)
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
            sleep(Duration::from_millis(20));
        }

        assert!(
            connected,
            "timed out waiting for k3-kv socket to become ready at {}",
            socket_path.display()
        );

        Self {
            socket_path,
            process: Some(child),
        }
    }

    fn stop_with_signal(&mut self, signal: libc::c_int) -> i32 {
        if let Some(mut child) = self.process.take() {
            let pid = child.id() as libc::pid_t;
            // SAFETY: libc::kill sends signal to child process.
            unsafe {
                libc::kill(pid, signal);
            }
            let status = child.wait().expect("failed to wait on daemon");
            status.code().unwrap_or(-1)
        } else {
            0
        }
    }
}

impl Drop for DaemonHarness {
    fn drop(&mut self) {
        if let Some(mut child) = self.process.take() {
            let pid = child.id() as libc::pid_t;
            // SAFETY: libc::kill sends SIGKILL to child process.
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
            let _ = child.wait();
        }
        if self.socket_path.exists() {
            let _ = fs::remove_file(&self.socket_path);
        }
    }
}

#[test]
fn test_cli_invalid_arguments_return_exit_code_2() {
    let bin = env!("CARGO_BIN_EXE_k3-kv");

    // Case 1: Unrecognized flag
    let out1 = Command::new(bin)
        .arg("--unrecognized-option")
        .output()
        .expect("run binary");
    assert_eq!(out1.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out1.stderr).contains("unknown option"));

    // Case 2: Missing argument to --socket
    let out2 = Command::new(bin)
        .arg("--socket")
        .output()
        .expect("run binary");
    assert_eq!(out2.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out2.stderr).contains("--socket requires a file path"));

    // Case 3: Empty string to --socket
    let out3 = Command::new(bin)
        .arg("--socket")
        .arg("   ")
        .output()
        .expect("run binary");
    assert_eq!(out3.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out3.stderr).contains("--socket requires a non-empty file path")
    );

    // Case 4: Non-numeric --max-memory-bytes
    let out4 = Command::new(bin)
        .arg("--max-memory-bytes")
        .arg("not-a-number")
        .output()
        .expect("run binary");
    assert_eq!(out4.status.code(), Some(2));

    // Case 5: Zero --max-memory-bytes
    let out5 = Command::new(bin)
        .arg("--max-memory-bytes")
        .arg("0")
        .output()
        .expect("run binary");
    assert_eq!(out5.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out5.stderr).contains("must be greater than 0"));
}

#[test]
fn test_cli_help_and_version_return_exit_code_0() {
    let bin = env!("CARGO_BIN_EXE_k3-kv");

    let out_help = Command::new(bin).arg("--help").output().expect("run help");
    assert_eq!(out_help.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out_help.stdout).contains("USAGE:"));

    let out_ver = Command::new(bin)
        .arg("--version")
        .output()
        .expect("run version");
    assert_eq!(out_ver.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&out_ver.stdout).contains("k3-kv"));
}

#[test]
fn test_daemon_startup_ping_and_sigterm_shutdown() {
    let socket = generate_test_socket_path();
    let mut harness = DaemonHarness::start(socket.clone(), None);

    // Verify client ping
    let stream = UnixStream::connect(&socket).expect("connect to daemon");
    let ping = IpcMessage::PingRequest(PingRequest {
        timestamp_ns: 1_700_000_123_456,
    });
    send_msg_frame(&stream, &ping, None).expect("send ping");

    let (response, opt_fd) = recv_msg_frame(&stream).expect("receive ping response");
    assert!(opt_fd.is_none());
    match response {
        IpcMessage::PingResponse(resp) => {
            assert_eq!(resp.timestamp_ns, 1_700_000_123_456);
        }
        other => panic!("expected PingResponse, got {other:?}"),
    }

    drop(stream);

    // Graceful shutdown via SIGTERM
    let exit_code = harness.stop_with_signal(libc::SIGTERM);
    assert_eq!(exit_code, 0, "daemon should exit with code 0 on SIGTERM");

    // Socket file must be removed from filesystem
    assert!(
        !socket.exists(),
        "socket file should have been unlinked on shutdown"
    );
}

#[test]
fn test_daemon_arena_allocation_scm_rights_reception_and_mutation() {
    let socket = generate_test_socket_path();
    let mut harness = DaemonHarness::start(socket.clone(), Some(1024 * 1024 * 1024)); // 1 GiB limit

    let stream = UnixStream::connect(&socket).expect("connect client 1");

    // 1. Allocate arena
    let alloc_req = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
        model_id: "qwen2.5-7b-instruct".to_owned(),
        arena_bytes: 32 * 1024 * 1024, // 32 MiB
        n_ctx: 512,
        max_blocks: 32,
    });
    send_msg_frame(&stream, &alloc_req, None).expect("send alloc request");

    let (alloc_resp, _) = recv_msg_frame(&stream).expect("recv alloc response");
    let arena_id = match alloc_resp {
        IpcMessage::AllocateArenaResponse(resp) => {
            assert_eq!(resp.status, StatusCode::Success);
            assert!(resp.arena_id > 0);
            resp.arena_id
        }
        other => panic!("expected AllocateArenaResponse, got {other:?}"),
    };

    // 2. Request arena FD via SCM_RIGHTS
    let get_fd_req = IpcMessage::GetArenaFdRequest(GetArenaFdRequest { arena_id });
    send_msg_frame(&stream, &get_fd_req, None).expect("send get_fd request");

    let (get_fd_resp, opt_fd) = recv_msg_frame(&stream).expect("recv get_fd response");
    match get_fd_resp {
        IpcMessage::GetArenaFdResponse(resp) => {
            assert_eq!(resp.status, StatusCode::Success);
        }
        other => panic!("expected GetArenaFdResponse, got {other:?}"),
    }

    let raw_fd = opt_fd.expect("expected SCM_RIGHTS FD passed with response");

    // 3. Map arena in client address space
    let mut client_arena =
        SharedMemoryArena::from_existing_fd(raw_fd, 32 * 1024 * 1024, "qwen2.5-7b-instruct", false)
            .expect("map existing shm fd in client");

    // Verify ArenaHeader initialized by KvHolder
    {
        let header = client_arena.header().expect("read header");
        assert_eq!(header.magic, ARENA_MAGIC);
        assert_eq!(header.version, ARENA_VERSION);
        assert_eq!(header.n_ctx, 512);
        assert_eq!(header.block_count, 32);
    }

    // 4. Mutate shared memory arena in-place
    {
        let meta = client_arena
            .block_meta_mut(0)
            .expect("get mutable block meta 0");
        meta.block_id = 101;
        meta.tag = TAG_PERSIST;
        meta.committed_token_count = 16;
        meta.sampler_rng_state = [0x1234_5678_9ABC_DEF0, 0, 0, 0];

        // Mutate arbitrary data byte
        let slice = client_arena.as_mut_slice();
        slice[2048] = 0xFE;
        slice[2049] = 0xED;
    }

    // 5. Connect client 2 and retrieve FD for the SAME arena
    let stream2 = UnixStream::connect(&socket).expect("connect client 2");
    send_msg_frame(&stream2, &get_fd_req, None).expect("send get_fd request from client 2");
    let (_, opt_fd2) = recv_msg_frame(&stream2).expect("recv get_fd from client 2");
    let raw_fd2 = opt_fd2.expect("expected SCM_RIGHTS FD for client 2");

    let second_client_arena = SharedMemoryArena::from_existing_fd(
        raw_fd2,
        32 * 1024 * 1024,
        "qwen2.5-7b-instruct",
        true, // read-only view
    )
    .expect("map existing fd in client 2");

    // Verify client 2 observes in-place mutations without copying
    {
        let meta2 = second_client_arena
            .block_meta(0)
            .expect("read block meta 0 from client 2");
        assert_eq!(meta2.block_id, 101);
        assert_eq!(meta2.tag, TAG_PERSIST);
        assert_eq!(meta2.committed_token_count, 16);
        assert_eq!(meta2.sampler_rng_state, [0x1234_5678_9ABC_DEF0, 0, 0, 0]);

        let slice2 = second_client_arena.as_slice();
        assert_eq!(slice2[2048], 0xFE);
        assert_eq!(slice2[2049], 0xED);
    }

    // 6. Deallocate arena
    let dealloc_req = IpcMessage::DeallocateArenaRequest(DeallocateArenaRequest { arena_id });
    send_msg_frame(&stream, &dealloc_req, None).expect("send dealloc request");
    let (dealloc_resp, _) = recv_msg_frame(&stream).expect("recv dealloc response");
    match dealloc_resp {
        IpcMessage::DeallocateArenaResponse(resp) => {
            assert_eq!(resp.status, StatusCode::Success);
            assert_eq!(resp.arena_id, arena_id);
        }
        other => panic!("expected DeallocateArenaResponse, got {other:?}"),
    }

    // 7. Verify subsequent lookup fails with ArenaNotFound
    send_msg_frame(&stream, &get_fd_req, None).expect("lookup deallocated arena");
    let (not_found_resp, opt_fd_none) = recv_msg_frame(&stream).expect("recv response");
    assert!(opt_fd_none.is_none());
    match not_found_resp {
        IpcMessage::GetArenaFdResponse(resp) => {
            assert_eq!(resp.status, StatusCode::ArenaNotFound);
        }
        other => panic!("expected ArenaNotFound, got {other:?}"),
    }

    drop(stream);
    drop(stream2);
    drop(client_arena);
    drop(second_client_arena);

    // Graceful stop with SIGINT
    let exit_code = harness.stop_with_signal(libc::SIGINT);
    assert_eq!(exit_code, 0);
    assert!(!socket.exists());
}

#[test]
fn test_daemon_memlock_ceiling_rejection() {
    let socket = generate_test_socket_path();
    // Start daemon with small 64 MiB ceiling
    let mut harness = DaemonHarness::start(socket.clone(), Some(64 * 1024 * 1024));

    let stream = UnixStream::connect(&socket).expect("connect client");

    // Request 128 MiB (exceeds 64 MiB ceiling)
    let oversize_req = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
        model_id: "oversized-model".to_owned(),
        arena_bytes: 128 * 1024 * 1024,
        n_ctx: 1024,
        max_blocks: 64,
    });
    send_msg_frame(&stream, &oversize_req, None).expect("send oversized request");

    let (response, _) = recv_msg_frame(&stream).expect("recv response");
    match response {
        IpcMessage::AllocateArenaResponse(resp) => {
            assert_eq!(resp.status, StatusCode::InsufficientLockableMemory);
            assert_eq!(resp.arena_id, 0);
            assert!(resp.error_message.is_some());
        }
        other => panic!("expected AllocateArenaResponse, got {other:?}"),
    }

    // Verify daemon is still alive and responds to ping
    let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 42 });
    send_msg_frame(&stream, &ping, None).expect("send ping");
    let (ping_resp, _) = recv_msg_frame(&stream).expect("recv ping");
    assert_eq!(
        ping_resp,
        IpcMessage::PingResponse(PingResponse { timestamp_ns: 42 })
    );

    drop(stream);

    harness.stop_with_signal(libc::SIGTERM);
}
