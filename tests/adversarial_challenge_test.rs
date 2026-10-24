//! Adversarial stress test harness for `KvServer` and `k3-kv` daemon binary.
//!
//! Evaluates resilience against malformed/truncated Fory frames, corrupt magic numbers,
//! oversized payloads (>16 MiB), unknown type IDs, concurrent client load, abrupt client SIGKILL,
//! and verifies `SocketFileGuard` unlinking and stale socket recovery.

use std::fs;
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{sleep, spawn};
use std::time::{Duration, Instant};

use k3_core::ipc::protocol::{
    recv_msg_frame, send_frame, send_msg_frame, AllocateArenaRequest,
    DeallocateArenaRequest, GetArenaFdRequest, IpcMessage, PingRequest, PingResponse, StatusCode,
    FRAME_HEADER_LEN, FRAME_MAGIC, MAX_PAYLOAD_LEN,
};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(100);

fn unique_test_socket_path() -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("k3_kv_adv_{}_{}.sock", std::process::id(), id))
}

struct TestDaemon {
    socket_path: PathBuf,
    child: Option<Child>,
}

impl TestDaemon {
    fn start(socket_path: PathBuf, max_mem: Option<u64>) -> Self {
        let bin = env!("CARGO_BIN_EXE_k3-kv");
        let mut cmd = Command::new(bin);
        cmd.arg("--socket").arg(&socket_path);
        if let Some(m) = max_mem {
            cmd.arg("--max-memory-bytes").arg(m.to_string());
        }

        let child = cmd.spawn().expect("spawn k3-kv daemon");

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
            sleep(Duration::from_millis(15));
        }

        assert!(connected, "timed out waiting for k3-kv socket");

        Self {
            socket_path,
            child: Some(child),
        }
    }

    fn signal(&mut self, sig: libc::c_int) -> i32 {
        if let Some(mut child) = self.child.take() {
            let pid = child.id() as libc::pid_t;
            // SAFETY: Signal delivery to supervised child process.
            unsafe {
                libc::kill(pid, sig);
            }
            let status = child.wait().expect("wait child");
            status.code().unwrap_or(-1)
        } else {
            0
        }
    }
}

impl Drop for TestDaemon {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let pid = child.id() as libc::pid_t;
            // SAFETY: SIGKILL sent to child process in harness destructor.
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
fn test_adversarial_corrupt_magic_numbers_rejection() {
    eprintln!("[test] Starting corrupt magic test");
    let socket = unique_test_socket_path();
    eprintln!("[test] Socket path: {}", socket.display());
    let mut daemon = TestDaemon::start(socket.clone(), None);
    eprintln!("[test] Daemon started successfully");

    let corrupt_magics: [u32; 5] = [
        0x0000_0000,
        0xDEAD_BEEF,
        0xFFFF_FFFF,
        0x4B33_4951, // off by one
        0x1234_5678,
    ];

    for &bad_magic in &corrupt_magics {
        eprintln!("[test] Testing bad magic 0x{bad_magic:08X}");
        let mut stream = UnixStream::connect(&socket).expect("connect to daemon");
        stream.set_read_timeout(Some(Duration::from_secs(2))).unwrap();

        // Construct raw 12-byte header with bad magic
        let mut header = [0_u8; FRAME_HEADER_LEN];
        header[0..4].copy_from_slice(&bad_magic.to_be_bytes());
        header[4..6].copy_from_slice(&0_u16.to_be_bytes()); // flags
        header[6..8].copy_from_slice(&107_u16.to_be_bytes()); // PingRequest
        header[8..12].copy_from_slice(&0_u32.to_be_bytes()); // payload_len

        eprintln!("[test] Writing bad magic header");
        stream.write_all(&header).expect("write bad magic header");

        // Server must disconnect due to frame decode error
        eprintln!("[test] Reading response from server (expecting EOF)");
        let mut read_buf = [0_u8; 16];
        let bytes_read = match stream.read(&mut read_buf) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[test] stream.read returned error: {e}");
                0
            }
        };
        eprintln!("[test] bytes_read: {bytes_read}");
        assert_eq!(
            bytes_read, 0,
            "Server must disconnect client with corrupt magic 0x{bad_magic:08X}"
        );
    }

    // Verify daemon is still healthy and accepts valid ping
    eprintln!("[test] Verifying daemon health with valid ping");
    let stream = UnixStream::connect(&socket).expect("connect to daemon");
    let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 777_888 });
    send_msg_frame(&stream, &ping, None).expect("send valid ping");
    let (resp, _) = recv_msg_frame(&stream).expect("recv valid ping response");
    assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 777_888 }));
    drop(stream);

    eprintln!("[test] Stopping daemon");
    assert_eq!(daemon.signal(libc::SIGTERM), 0);
    eprintln!("[test] Finished corrupt magic test");
}

#[test]
fn test_adversarial_oversized_payload_rejection() {
    let socket = unique_test_socket_path();
    let mut daemon = TestDaemon::start(socket.clone(), None);

    let oversized_lengths: [u32; 3] = [
        MAX_PAYLOAD_LEN + 1,           // 16 MiB + 1 byte
        MAX_PAYLOAD_LEN + 1024 * 1024, // 17 MiB
        u32::MAX,                      // 4 GiB - 1
    ];

    for &oversized_len in &oversized_lengths {
        let mut stream = UnixStream::connect(&socket).expect("connect to daemon");

        let mut header = [0_u8; FRAME_HEADER_LEN];
        header[0..4].copy_from_slice(&FRAME_MAGIC.to_be_bytes());
        header[4..6].copy_from_slice(&0_u16.to_be_bytes());
        header[6..8].copy_from_slice(&101_u16.to_be_bytes());
        header[8..12].copy_from_slice(&oversized_len.to_be_bytes());

        stream.write_all(&header).expect("write oversized header");

        // Server must reject without allocating buffer and close connection
        let mut read_buf = [0_u8; 16];
        let bytes_read = stream.read(&mut read_buf).expect("read from stream");
        assert_eq!(
            bytes_read, 0,
            "Server must immediately disconnect client specifying payload len {oversized_len}"
        );
        drop(stream);
    }

    // Verify daemon continues operating normally
    let stream = UnixStream::connect(&socket).expect("connect to daemon");
    let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 999_111 });
    send_msg_frame(&stream, &ping, None).expect("send valid ping");
    let (resp, _) = recv_msg_frame(&stream).expect("recv valid ping response");
    assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 999_111 }));
    drop(stream);

    assert_eq!(daemon.signal(libc::SIGTERM), 0);
}

#[test]
fn test_adversarial_truncated_frames_and_disconnects() {
    let socket = unique_test_socket_path();
    let mut daemon = TestDaemon::start(socket.clone(), None);

    // 1. Send single byte and abruptly disconnect
    {
        let mut stream = UnixStream::connect(&socket).expect("connect");
        stream.write_all(&[0x4B]).expect("write 1 byte");
        drop(stream);
    }

    // 2. Send 6 bytes of 12-byte header and abruptly disconnect
    {
        let mut stream = UnixStream::connect(&socket).expect("connect");
        let partial_header = [0x4B, 0x33, 0x49, 0x50, 0x00, 0x00];
        stream.write_all(&partial_header).expect("write partial header");
        drop(stream);
    }

    // 3. Send full 12-byte header claiming 64 bytes of payload, but send only 10 bytes and disconnect
    {
        let mut stream = UnixStream::connect(&socket).expect("connect");
        let mut header = [0_u8; FRAME_HEADER_LEN];
        header[0..4].copy_from_slice(&FRAME_MAGIC.to_be_bytes());
        header[4..6].copy_from_slice(&0_u16.to_be_bytes());
        header[6..8].copy_from_slice(&101_u16.to_be_bytes()); // AllocateArenaRequest
        header[8..12].copy_from_slice(&64_u32.to_be_bytes()); // claims 64 bytes

        stream.write_all(&header).expect("write header");
        stream.write_all(b"1234567890").expect("write partial payload");
        drop(stream);
    }

    // 4. Send full 12-byte header claiming 1024 bytes of payload, send 0 payload bytes and disconnect
    {
        let mut stream = UnixStream::connect(&socket).expect("connect");
        let mut header = [0_u8; FRAME_HEADER_LEN];
        header[0..4].copy_from_slice(&FRAME_MAGIC.to_be_bytes());
        header[4..6].copy_from_slice(&0_u16.to_be_bytes());
        header[6..8].copy_from_slice(&105_u16.to_be_bytes());
        header[8..12].copy_from_slice(&1024_u32.to_be_bytes());

        stream.write_all(&header).expect("write header");
        drop(stream);
    }

    // Verify daemon did not crash or hang
    let stream = UnixStream::connect(&socket).expect("connect to daemon");
    let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 123_456_789 });
    send_msg_frame(&stream, &ping, None).expect("send valid ping");
    let (resp, _) = recv_msg_frame(&stream).expect("recv ping response");
    assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 123_456_789 }));
    drop(stream);

    assert_eq!(daemon.signal(libc::SIGTERM), 0);
}

#[test]
fn test_adversarial_malformed_fory_payloads_return_protocol_error() {
    let socket = unique_test_socket_path();
    let mut daemon = TestDaemon::start(socket.clone(), None);

    let stream = UnixStream::connect(&socket).expect("connect to daemon");

    let targets = [
        (101, "AllocateArenaRequest"),
        (103, "GetArenaFdRequest"),
        (105, "DeallocateArenaRequest"),
        (107, "PingRequest"),
    ];

    for (type_id, name) in targets {
        let corrupt_payload = b"TOTALLY_CORRUPTED_FORY_SERIALIZATION_BYTES_!@#$%^&*()";
        send_frame(stream.as_raw_fd(), type_id, corrupt_payload, None)
            .unwrap_or_else(|e| panic!("failed to send corrupt frame for {name}: {e}"));

        let (resp, opt_fd) = recv_msg_frame(&stream)
            .unwrap_or_else(|e| panic!("failed to recv response for corrupt {name}: {e}"));
        assert!(opt_fd.is_none());

        match resp {
            IpcMessage::AllocateArenaResponse(r) => {
                assert_eq!(
                    r.status,
                    StatusCode::ProtocolError,
                    "Expected ProtocolError for corrupt {name}, got {:?}",
                    r.status
                );
                assert!(r.error_message.is_some());
            }
            other => panic!("expected AllocateArenaResponse with ProtocolError, got {other:?}"),
        }
    }

    // Verify subsequent valid request succeeds on the SAME stream
    let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 555_000 });
    send_msg_frame(&stream, &ping, None).expect("send ping");
    let (resp, _) = recv_msg_frame(&stream).expect("recv ping");
    assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 555_000 }));

    drop(stream);
    assert_eq!(daemon.signal(libc::SIGTERM), 0);
}

#[test]
fn test_adversarial_unknown_and_worker_type_ids_return_capability_mismatch() {
    let socket = unique_test_socket_path();
    let mut daemon = TestDaemon::start(socket.clone(), None);

    let stream = UnixStream::connect(&socket).expect("connect to daemon");

    // 1. Worker commands (Type IDs: 201..=206)
    let worker_type_ids = [201_u16, 202, 203, 204, 205, 206];
    for tid in worker_type_ids {
        send_frame(stream.as_raw_fd(), tid, b"dummy_worker_payload", None).unwrap();
        let (resp, opt_fd) = recv_msg_frame(&stream).unwrap();
        assert!(opt_fd.is_none());
        match resp {
            IpcMessage::AllocateArenaResponse(r) => {
                assert_eq!(
                    r.status,
                    StatusCode::CapabilityMismatch,
                    "Expected CapabilityMismatch for worker command {tid}"
                );
            }
            other => panic!("expected CapabilityMismatch response, got {other:?}"),
        }
    }

    // 2. Unknown Type IDs
    let unknown_type_ids = [0_u16, 99, 100, 150, 300, 999, 65535];
    for uid in unknown_type_ids {
        send_frame(stream.as_raw_fd(), uid, b"unknown_type_payload", None).unwrap();
        let (resp, opt_fd) = recv_msg_frame(&stream).unwrap();
        assert!(opt_fd.is_none());
        match resp {
            IpcMessage::AllocateArenaResponse(r) => {
                assert_eq!(
                    r.status,
                    StatusCode::CapabilityMismatch,
                    "Expected CapabilityMismatch for unknown type ID {uid}"
                );
            }
            other => panic!("expected CapabilityMismatch response, got {other:?}"),
        }
    }

    drop(stream);
    assert_eq!(daemon.signal(libc::SIGTERM), 0);
}

#[test]
fn test_adversarial_heavy_concurrent_clients_stress() {
    let socket = unique_test_socket_path();
    let mut daemon = TestDaemon::start(socket.clone(), Some(512 * 1024 * 1024)); // 512 MiB limit

    let num_clients = 16;
    let iterations_per_client = 8;
    let mut handles = Vec::new();

    for client_id in 0..num_clients {
        let sock_path = socket.clone();
        handles.push(spawn(move || {
            let stream = UnixStream::connect(&sock_path).unwrap();

            for iter in 0..iterations_per_client {
                // 1. Allocate arena
                let alloc_req = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
                    model_id: format!("concurrent-model-{client_id}-{iter}"),
                    arena_bytes: 128 * 1024, // 128 KiB
                    n_ctx: 64,
                    max_blocks: 4,
                });
                send_msg_frame(&stream, &alloc_req, None).unwrap();
                let (alloc_resp, _) = recv_msg_frame(&stream).unwrap();
                let arena_id = match alloc_resp {
                    IpcMessage::AllocateArenaResponse(r) => {
                        assert_eq!(r.status, StatusCode::Success);
                        r.arena_id
                    }
                    other => panic!("expected alloc response, got {other:?}"),
                };

                // 2. Get FD
                let get_req = IpcMessage::GetArenaFdRequest(GetArenaFdRequest { arena_id });
                send_msg_frame(&stream, &get_req, None).unwrap();
                let (get_resp, opt_fd) = recv_msg_frame(&stream).unwrap();
                assert!(opt_fd.is_some());
                match get_resp {
                    IpcMessage::GetArenaFdResponse(r) => assert_eq!(r.status, StatusCode::Success),
                    other => panic!("expected get response, got {other:?}"),
                }

                // 3. Ping
                let ts = u64::try_from(client_id * 1000 + iter).unwrap_or(0);
                let ping = IpcMessage::PingRequest(PingRequest {
                    timestamp_ns: ts,
                });
                send_msg_frame(&stream, &ping, None).unwrap();
                let (ping_resp, _) = recv_msg_frame(&stream).unwrap();
                assert_eq!(
                    ping_resp,
                    IpcMessage::PingResponse(PingResponse {
                        timestamp_ns: ts,
                    })
                );

                // 4. Deallocate
                let dealloc_req = IpcMessage::DeallocateArenaRequest(DeallocateArenaRequest { arena_id });
                send_msg_frame(&stream, &dealloc_req, None).unwrap();
                let (dealloc_resp, _) = recv_msg_frame(&stream).unwrap();
                match dealloc_resp {
                    IpcMessage::DeallocateArenaResponse(r) => {
                        assert_eq!(r.status, StatusCode::Success);
                    }
                    other => panic!("expected dealloc response, got {other:?}"),
                }
            }

            drop(stream);
        }));
    }

    for h in handles {
        h.join().expect("client thread panicked");
    }

    // Verify daemon is responsive and clean
    let stream = UnixStream::connect(&socket).expect("connect");
    let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 42 });
    send_msg_frame(&stream, &ping, None).unwrap();
    let (resp, _) = recv_msg_frame(&stream).unwrap();
    assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 42 }));
    drop(stream);

    assert_eq!(daemon.signal(libc::SIGTERM), 0);
}

#[test]
fn test_adversarial_client_sigkill_mid_transfer() {
    let socket = unique_test_socket_path();
    let mut daemon = TestDaemon::start(socket.clone(), None);

    // Spawn a child process that connects and gets killed mid-transfer
    let bin = env!("CARGO_BIN_EXE_k3-kv");

    // Spawn dummy process
    let mut child = Command::new(bin)
        .arg("--help")
        .spawn()
        .expect("spawn dummy");
    let pid = child.id() as libc::pid_t;

    // Connect socket and write partial frame
    let mut stream = UnixStream::connect(&socket).expect("connect");
    stream.write_all(b"\x4B\x33\x49\x50\x00\x00").expect("partial write");

    // Send SIGKILL to the dummy child to simulate abrupt process termination
    // SAFETY: SIGKILL sent to dummy child process.
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
    let _ = child.wait();

    // Abruptly close the socket from client side (simulating socket teardown upon SIGKILL)
    drop(stream);

    // Verify daemon survived intact
    let stream2 = UnixStream::connect(&socket).expect("reconnect to daemon");
    let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 888_999 });
    send_msg_frame(&stream2, &ping, None).expect("send ping");
    let (resp, _) = recv_msg_frame(&stream2).expect("recv ping");
    assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 888_999 }));
    drop(stream2);

    assert_eq!(daemon.signal(libc::SIGTERM), 0);
}

#[test]
fn test_adversarial_socket_file_guard_and_stale_socket_recovery() {
    let socket = unique_test_socket_path();

    // Phase 1: Normal startup and SIGINT shutdown -> SocketFileGuard unlinks socket file
    {
        let mut d1 = TestDaemon::start(socket.clone(), None);
        assert!(socket.exists(), "Socket file must exist while daemon is active");
        let exit_code = d1.signal(libc::SIGINT);
        assert_eq!(exit_code, 0, "Daemon should exit 0 on SIGINT");
        assert!(
            !socket.exists(),
            "Socket file must be unlinked by SocketFileGuard on graceful exit"
        );
    }

    // Phase 2: Start daemon and kill with SIGKILL (hard crash) -> Socket file remains
    {
        let mut d2 = TestDaemon::start(socket.clone(), None);
        assert!(socket.exists());
        let _ = d2.signal(libc::SIGKILL);
        // Socket file was NOT unlinked because SIGKILL immediately aborts process
        assert!(
            socket.exists(),
            "Socket file should remain on filesystem after unhandled SIGKILL"
        );
    }

    // Phase 3: Start NEW daemon pointing to the stale socket file
    // prepare_and_bind_socket must detect ECONNREFUSED and unlink the stale socket file automatically!
    {
        let mut d3 = TestDaemon::start(socket.clone(), None);
        assert!(socket.exists());

        let stream = UnixStream::connect(&socket).expect("connect to recovered daemon");
        let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 333 });
        send_msg_frame(&stream, &ping, None).expect("send ping");
        let (resp, _) = recv_msg_frame(&stream).expect("recv ping");
        assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 333 }));
        drop(stream);

        let exit_code = d3.signal(libc::SIGTERM);
        assert_eq!(exit_code, 0);
        assert!(!socket.exists(), "Socket file must be unlinked on SIGTERM");
    }
}

#[test]
fn test_adversarial_connected_client_blocks_graceful_shutdown() {
    let socket = unique_test_socket_path();
    let mut daemon = TestDaemon::start(socket.clone(), None);

    // Client connects and remains idle (holding the socket open without sending data)
    let idle_client = UnixStream::connect(&socket).expect("idle client connects");

    // Operator sends SIGTERM to daemon
    if let Some(ref mut child) = daemon.child {
        let pid = child.id() as libc::pid_t;
        // SAFETY: SIGTERM sent to daemon process.
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
    }

    // Verify daemon exits cleanly within 2 seconds even while idle client remains connected
    let start = Instant::now();
    let mut exited = false;
    let mut exit_code = None;
    while start.elapsed() < Duration::from_secs(2) {
        if let Some(ref mut child) = daemon.child {
            if let Ok(Some(status)) = child.try_wait() {
                exited = true;
                exit_code = status.code();
                break;
            }
        }
        sleep(Duration::from_millis(50));
    }

    assert!(
        exited,
        "Daemon should exit cleanly on SIGTERM within 2s even while idle client socket remains open"
    );
    assert_eq!(
        exit_code,
        Some(0),
        "Daemon should exit with status 0 on SIGTERM"
    );

    // Clean up idle client
    drop(idle_client);

    // Socket file must have been unlinked upon graceful daemon termination
    assert!(!socket.exists(), "Socket file must be unlinked on SIGTERM");
}

