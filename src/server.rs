//! Unix Domain Socket server and message dispatch loop for `k3-kv`.

use std::fs;
use std::io;
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use k3_core::ipc::protocol::{
    global_fory, recv_frame, send_msg_frame, AllocateArenaRequest, AllocateArenaResponse,
    DeallocateArenaRequest, DeallocateArenaResponse, GetArenaFdRequest, GetArenaFdResponse,
    IpcMessage, PingRequest, PingResponse, StatusCode,
};
use k3_core::ipc::shm::DEFAULT_LIMIT_MEMLOCK_BYTES;

use crate::holder::KvHolder;

/// Default Unix domain socket path for `k3-kv`.
pub const DEFAULT_SOCKET_PATH: &str = "/tmp/k3-kv.sock";

/// Configuration for [`KvServer`].
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Filesystem path to the Unix domain socket.
    pub socket_path: PathBuf,
    /// Bounded memlock ceiling in bytes (defaults to 28 GiB, AGENTS.md Rule 5).
    pub max_memory_bytes: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from(DEFAULT_SOCKET_PATH),
            max_memory_bytes: DEFAULT_LIMIT_MEMLOCK_BYTES,
        }
    }
}

/// RAII guard that automatically removes the Unix domain socket file upon drop.
#[derive(Debug)]
pub struct SocketFileGuard {
    path: PathBuf,
    active: bool,
}

impl SocketFileGuard {
    /// Creates a new guard for the given socket file path.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self { path, active: true }
    }

    /// Disarms the guard so the socket file is retained on drop.
    pub fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for SocketFileGuard {
    fn drop(&mut self) {
        if self.active && self.path.exists() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Prepares and binds a `tokio::net::UnixListener` after verifying and cleaning stale socket files.
///
/// # Errors
/// Returns [`io::Error`] if an active daemon is already listening or binding fails.
pub fn prepare_and_bind_socket(socket_path: &Path) -> io::Result<tokio::net::UnixListener> {
    if let Some(parent) = socket_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }

    if socket_path.exists() {
        // Safe state verification (AGENTS.md Rule 1): probe if another daemon is alive
        match UnixStream::connect(socket_path) {
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!("Another k3-kv daemon is actively listening on {}", socket_path.display()),
                ));
            }
            Err(e) if e.kind() == io::ErrorKind::ConnectionRefused => {
                // Stale socket file from previous unclean exit -> unlink it
                let _ = fs::remove_file(socket_path);
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // Socket file disappeared concurrently -> proceed
            }
            Err(_) => {
                // Broken or unconnectable node -> attempt unlink
                let _ = fs::remove_file(socket_path);
            }
        }
    }

    let listener = tokio::net::UnixListener::bind(socket_path)?;

    // Set secure socket permissions (0600: read/write by owner only)
    #[cfg(unix)]
    {
        let perms = fs::Permissions::from_mode(0o600);
        let _ = fs::set_permissions(socket_path, perms);
    }

    Ok(listener)
}

/// Standalone Unix Domain Socket server managing KV cache arenas.
#[derive(Debug)]
pub struct KvServer {
    config: ServerConfig,
    holder: Arc<KvHolder>,
    shutdown_notify: Arc<tokio::sync::Notify>,
    is_shutdown: Arc<AtomicBool>,
    active_clients: Arc<Mutex<HashMap<u64, UnixStream>>>,
    next_client_id: Arc<AtomicU64>,
}

impl KvServer {
    /// Creates a new [`KvServer`] with the specified configuration.
    #[must_use]
    pub fn new(config: ServerConfig) -> Self {
        let holder = Arc::new(KvHolder::with_memlock_ceiling(config.max_memory_bytes));
        Self::with_holder(config, holder)
    }

    /// Creates a new [`KvServer`] wrapping an existing [`KvHolder`].
    #[must_use]
    pub fn with_holder(config: ServerConfig, holder: Arc<KvHolder>) -> Self {
        Self {
            config,
            holder,
            shutdown_notify: Arc::new(tokio::sync::Notify::new()),
            is_shutdown: Arc::new(AtomicBool::new(false)),
            active_clients: Arc::new(Mutex::new(HashMap::new())),
            next_client_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Creates a new [`KvServer`] bound to a given path with the specified holder.
    ///
    /// # Errors
    /// Returns [`io::Error`] if path verification fails.
    pub fn bind(socket_path: impl AsRef<Path>, holder: KvHolder) -> io::Result<Self> {
        let path = socket_path.as_ref().to_path_buf();
        let max_mem = holder.memlock_ceiling_bytes();
        Ok(Self::with_holder(
            ServerConfig {
                socket_path: path,
                max_memory_bytes: max_mem,
            },
            Arc::new(holder),
        ))
    }

    /// Returns a reference to the inner memory holder.
    #[must_use]
    pub fn holder(&self) -> &Arc<KvHolder> {
        &self.holder
    }

    /// Returns the configured socket path.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.config.socket_path
    }

    /// Returns a shutdown notification handle.
    #[must_use]
    pub fn shutdown_handle(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.shutdown_notify)
    }

    /// Returns the number of currently active client connections.
    #[must_use]
    pub fn active_client_count(&self) -> usize {
        self.active_clients
            .lock()
            .map_or(0, |c| c.len())
    }

    /// Triggers graceful server shutdown, stopping the accept loop and terminating active client streams.
    pub fn shutdown(&self) {
        self.is_shutdown.store(true, Ordering::SeqCst);
        self.shutdown_notify.notify_waiters();
        self.shutdown_notify.notify_one();
        if let Ok(mut clients) = self.active_clients.lock() {
            for stream in clients.values() {
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
            clients.clear();
        }
    }

    /// Runs the server accept loop until shutdown notification is received.
    ///
    /// # Errors
    /// Returns [`io::Error`] on fatal socket or listener failure.
    pub async fn run(&self) -> io::Result<()> {
        let listener = prepare_and_bind_socket(&self.config.socket_path)?;
        let _guard = SocketFileGuard::new(self.config.socket_path.clone());

        while !self.is_shutdown.load(Ordering::SeqCst) {
            tokio::select! {
                accept_res = listener.accept() => {
                    match accept_res {
                        Ok((stream, _addr)) => {
                            if self.is_shutdown.load(Ordering::SeqCst) {
                                break;
                            }
                            let holder = Arc::clone(&self.holder);
                            match stream.into_std() {
                                Ok(std_stream) => {
                                    if let Err(e) = std_stream.set_nonblocking(false) {
                                        eprintln!("[k3-kv] Failed to set stream blocking mode: {e}");
                                        continue;
                                    }

                                    let client_id = self.next_client_id.fetch_add(1, Ordering::Relaxed);
                                    let active_clients = Arc::clone(&self.active_clients);

                                    // Track connection for shutdown signaling
                                    if let Ok(cloned) = std_stream.try_clone() {
                                        if let Ok(mut clients) = active_clients.lock() {
                                            if self.is_shutdown.load(Ordering::SeqCst) {
                                                let _ = cloned.shutdown(std::net::Shutdown::Both);
                                                break;
                                            }
                                            clients.insert(client_id, cloned);
                                        }
                                    }

                                    tokio::task::spawn_blocking(move || {
                                        struct ClientGuard {
                                            id: u64,
                                            clients: Arc<Mutex<HashMap<u64, UnixStream>>>,
                                        }
                                        impl Drop for ClientGuard {
                                            fn drop(&mut self) {
                                                if let Ok(mut map) = self.clients.lock() {
                                                    map.remove(&self.id);
                                                }
                                            }
                                        }
                                        let _guard = ClientGuard {
                                            id: client_id,
                                            clients: active_clients,
                                        };
                                        let _ = handle_connection(&std_stream, &holder);
                                    });
                                }
                                Err(e) => {
                                    eprintln!("[k3-kv] Failed to convert tokio stream to std: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[k3-kv] Listener accept error: {e}");
                        }
                    }
                }
                () = self.shutdown_notify.notified() => {
                    break;
                }
            }
        }

        // Cleanly terminate active client connections so blocking worker threads unblock immediately
        self.shutdown();
        Ok(())
    }
}

/// Internal dispatch errors for protocol rejection.
#[derive(Debug)]
pub enum DispatchError {
    /// Worker command or unsupported functionality.
    CapabilityMismatch(String),
    /// Frame corruption or decoding error.
    ProtocolError(String),
}

/// Handles all requests from a single connected client session sequentially.
///
/// # Errors
/// Returns [`io::Error`] on fatal I/O failure.
pub fn handle_connection(
    stream: &UnixStream,
    holder: &Arc<KvHolder>,
) -> io::Result<()> {
    let fory = global_fory();

    loop {
        let (header, payload, opt_fd) = match recv_frame(stream.as_raw_fd()) {
            Ok(res) => res,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // Client closed socket cleanly
                break;
            }
            Err(e) => {
                // Invalid magic, payload > 16 MiB, or socket read failure
                return Err(e);
            }
        };

        // Drop any unexpected client-supplied FD safely
        drop(opt_fd);

        let dispatch_res = dispatch_single_request(header.type_id, &payload, holder, fory);
        match dispatch_res {
            Ok((resp_msg, pass_fd)) => {
                send_msg_frame(stream, &resp_msg, pass_fd)?;
            }
            Err(DispatchError::CapabilityMismatch(msg)) => {
                let err_resp = IpcMessage::AllocateArenaResponse(AllocateArenaResponse {
                    arena_id: 0,
                    arena_bytes: 0,
                    status: StatusCode::CapabilityMismatch,
                    error_message: Some(msg),
                });
                send_msg_frame(stream, &err_resp, None)?;
            }
            Err(DispatchError::ProtocolError(msg)) => {
                let err_resp = IpcMessage::AllocateArenaResponse(AllocateArenaResponse {
                    arena_id: 0,
                    arena_bytes: 0,
                    status: StatusCode::ProtocolError,
                    error_message: Some(msg),
                });
                send_msg_frame(stream, &err_resp, None)?;
            }
        }
    }

    Ok(())
}

/// Routes an individual incoming message to the appropriate `KvHolder` action.
///
/// # Errors
/// Returns [`DispatchError::ProtocolError`] if message deserialization fails, or
/// [`DispatchError::CapabilityMismatch`] for worker commands or unknown type IDs.
pub fn dispatch_single_request(
    type_id: u16,
    payload: &[u8],
    holder: &Arc<KvHolder>,
    fory: &fory::Fory,
) -> Result<(IpcMessage, Option<std::os::fd::RawFd>), DispatchError> {
    match type_id {
        // 1. AllocateArenaRequest (Type ID: 101)
        101 => {
            let req: AllocateArenaRequest = fory.deserialize(payload).map_err(|e| {
                DispatchError::ProtocolError(format!("Corrupt AllocateArenaRequest: {e}"))
            })?;

            let resp = match holder.allocate_arena(&req.model_id, req.arena_bytes, req.n_ctx, req.max_blocks) {
                Ok(res) => AllocateArenaResponse {
                    arena_id: res.arena_id,
                    arena_bytes: res.total_bytes,
                    status: StatusCode::Success,
                    error_message: None,
                },
                Err(err) => AllocateArenaResponse {
                    arena_id: 0,
                    arena_bytes: 0,
                    status: err.status_code(),
                    error_message: Some(err.to_string()),
                },
            };

            Ok((IpcMessage::AllocateArenaResponse(resp), None))
        }

        // 2. GetArenaFdRequest (Type ID: 103)
        103 => {
            let req: GetArenaFdRequest = fory.deserialize(payload).map_err(|e| {
                DispatchError::ProtocolError(format!("Corrupt GetArenaFdRequest: {e}"))
            })?;

            if let Some(arena) = holder.get_arena(req.arena_id) {
                let resp = GetArenaFdResponse {
                    arena_id: req.arena_id,
                    status: StatusCode::Success,
                    error_message: None,
                };
                let raw_fd = arena.as_raw_fd();
                Ok((IpcMessage::GetArenaFdResponse(resp), Some(raw_fd)))
            } else {
                let resp = GetArenaFdResponse {
                    arena_id: req.arena_id,
                    status: StatusCode::ArenaNotFound,
                    error_message: Some(format!("Arena {} not found", req.arena_id)),
                };
                Ok((IpcMessage::GetArenaFdResponse(resp), None))
            }
        }

        // 3. DeallocateArenaRequest (Type ID: 105)
        105 => {
            let req: DeallocateArenaRequest = fory.deserialize(payload).map_err(|e| {
                DispatchError::ProtocolError(format!("Corrupt DeallocateArenaRequest: {e}"))
            })?;

            let resp = match holder.deallocate(req.arena_id) {
                Ok(_) => DeallocateArenaResponse {
                    arena_id: req.arena_id,
                    status: StatusCode::Success,
                    error_message: None,
                },
                Err(err) => DeallocateArenaResponse {
                    arena_id: req.arena_id,
                    status: err.status_code(),
                    error_message: Some(err.to_string()),
                },
            };

            Ok((IpcMessage::DeallocateArenaResponse(resp), None))
        }

        // 4. PingRequest (Type ID: 107)
        107 => {
            let req: PingRequest = fory.deserialize(payload).map_err(|e| {
                DispatchError::ProtocolError(format!("Corrupt PingRequest: {e}"))
            })?;

            let resp = PingResponse {
                timestamp_ns: req.timestamp_ns,
            };

            Ok((IpcMessage::PingResponse(resp), None))
        }

        // 5. Compute Worker Commands sent to KV holder -> CapabilityMismatch (Exit code 10)
        201..=206 => Err(DispatchError::CapabilityMismatch(format!(
            "k3-kv is a context memory holder daemon and does not execute worker commands (type {type_id})"
        ))),

        // 6. Unknown Type ID
        unknown => Err(DispatchError::CapabilityMismatch(format!(
            "Unrecognized IPC message type ID: {unknown}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::ipc::layout::{ARENA_MAGIC, TAG_PERSIST};
    use k3_core::ipc::protocol::recv_msg_frame;
    use k3_core::ipc::shm::SharedMemoryArena;

    #[test]
    fn test_handle_connection_full_lifecycle() {
        let (server_fd, client_fd) = k3_core::ipc::protocol::create_uds_socket_pair().unwrap();
        let server_stream = UnixStream::from(server_fd);
        let client_stream = UnixStream::from(client_fd);

        let holder = Arc::new(KvHolder::with_memlock_ceiling(1024 * 1024 * 1024));
        let holder_clone = Arc::clone(&holder);

        let server_thread = std::thread::spawn(move || {
            handle_connection(&server_stream, &holder_clone).unwrap();
        });

        // 1. Ping
        let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 42_000_000 });
        send_msg_frame(&client_stream, &ping, None).unwrap();
        let (resp, _) = recv_msg_frame(&client_stream).unwrap();
        assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 42_000_000 }));

        // 2. AllocateArena
        let alloc = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
            model_id: "qwen2.5-7b".to_owned(),
            arena_bytes: 1024 * 1024,
            n_ctx: 128,
            max_blocks: 8,
        });
        send_msg_frame(&client_stream, &alloc, None).unwrap();
        let (resp_alloc, _) = recv_msg_frame(&client_stream).unwrap();
        let arena_id = match resp_alloc {
            IpcMessage::AllocateArenaResponse(r) => {
                assert_eq!(r.status, StatusCode::Success);
                assert!(r.arena_bytes >= 1024 * 1024);
                r.arena_id
            }
            other => panic!("Unexpected response: {other:?}"),
        };

        // 3. GetArenaFd with SCM_RIGHTS
        let get_fd = IpcMessage::GetArenaFdRequest(GetArenaFdRequest { arena_id });
        send_msg_frame(&client_stream, &get_fd, None).unwrap();
        let (resp_fd, opt_fd) = recv_msg_frame(&client_stream).unwrap();
        assert!(opt_fd.is_some(), "Must receive SCM_RIGHTS file descriptor");
        match resp_fd {
            IpcMessage::GetArenaFdResponse(r) => {
                assert_eq!(r.status, StatusCode::Success);
                assert_eq!(r.arena_id, arena_id);
            }
            other => panic!("Unexpected response: {other:?}"),
        }

        // 4. DeallocateArena
        let dealloc = IpcMessage::DeallocateArenaRequest(DeallocateArenaRequest { arena_id });
        send_msg_frame(&client_stream, &dealloc, None).unwrap();
        let (resp_dealloc, _) = recv_msg_frame(&client_stream).unwrap();
        match resp_dealloc {
            IpcMessage::DeallocateArenaResponse(r) => {
                assert_eq!(r.status, StatusCode::Success);
                assert_eq!(r.arena_id, arena_id);
            }
            other => panic!("Unexpected response: {other:?}"),
        }

        // 5. Close client and verify server finishes cleanly
        drop(client_stream);
        server_thread.join().unwrap();
    }

    #[test]
    fn test_scm_rights_fd_zero_copy_mutation() {
        let (server_fd, client_fd) = k3_core::ipc::protocol::create_uds_socket_pair().unwrap();
        let server_stream = UnixStream::from(server_fd);
        let client_stream = UnixStream::from(client_fd);

        let holder = Arc::new(KvHolder::with_memlock_ceiling(1024 * 1024 * 1024));
        let holder_clone = Arc::clone(&holder);

        let server_thread = std::thread::spawn(move || {
            handle_connection(&server_stream, &holder_clone).unwrap();
        });

        let alloc = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
            model_id: "mutation-test".to_owned(),
            arena_bytes: 256 * 1024,
            n_ctx: 64,
            max_blocks: 4,
        });
        send_msg_frame(&client_stream, &alloc, None).unwrap();
        let (resp_alloc, _) = recv_msg_frame(&client_stream).unwrap();
        let (arena_id, allocated_len) = match resp_alloc {
            IpcMessage::AllocateArenaResponse(r) => (r.arena_id, usize::try_from(r.arena_bytes).unwrap()),
            other => panic!("unexpected: {other:?}"),
        };

        let get_fd = IpcMessage::GetArenaFdRequest(GetArenaFdRequest { arena_id });
        send_msg_frame(&client_stream, &get_fd, None).unwrap();
        let (_, opt_fd) = recv_msg_frame(&client_stream).unwrap();
        let transferred_fd = opt_fd.expect("expected transferred FD");

        // Client maps the received FD
        let mut client_arena = SharedMemoryArena::from_existing_fd(
            transferred_fd,
            allocated_len,
            "mutation-test",
            false,
        ).unwrap();

        // Verify ArenaHeader signature
        let header = client_arena.header().unwrap();
        assert_eq!(header.magic, ARENA_MAGIC);

        // Client mutates KvBlockMeta table in place via client_arena
        let meta_mut = client_arena.block_meta_mut(0).unwrap();
        meta_mut.tag = TAG_PERSIST;
        meta_mut.committed_token_count = 16;
        meta_mut.sampler_rng_state = [0x1234_5678_9ABC_DEF0, 0, 0, 0];

        // Server-side holder reads its own mapped arena
        let server_arena = holder.get_arena(arena_id).unwrap();
        let server_meta = server_arena.block_meta(0).unwrap();

        // Verify shared-memory zero-copy coherence: server sees client mutation immediately
        assert_eq!(server_meta.tag, TAG_PERSIST);
        assert_eq!(server_meta.committed_token_count, 16);
        assert_eq!(server_meta.sampler_rng_state, [0x1234_5678_9ABC_DEF0, 0, 0, 0]);

        drop(client_stream);
        server_thread.join().unwrap();
    }

    #[test]
    fn test_protocol_memlock_ceiling_rejection() {
        let (server_fd, client_fd) = k3_core::ipc::protocol::create_uds_socket_pair().unwrap();
        let server_stream = UnixStream::from(server_fd);
        let client_stream = UnixStream::from(client_fd);

        // Bounded limit: 1 MiB max
        let holder = Arc::new(KvHolder::with_memlock_ceiling(1024 * 1024));
        let holder_clone = Arc::clone(&holder);

        let server_thread = std::thread::spawn(move || {
            handle_connection(&server_stream, &holder_clone).unwrap();
        });

        // Request 100 MiB (exceeds 1 MiB limit)
        let alloc = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
            model_id: "oversized".to_owned(),
            arena_bytes: 100 * 1024 * 1024,
            n_ctx: 1024,
            max_blocks: 64,
        });
        send_msg_frame(&client_stream, &alloc, None).unwrap();
        let (resp, _) = recv_msg_frame(&client_stream).unwrap();
        match resp {
            IpcMessage::AllocateArenaResponse(r) => {
                assert_eq!(r.status, StatusCode::InsufficientLockableMemory);
                assert_eq!(r.arena_id, 0);
                assert!(r.error_message.is_some());
            }
            other => panic!("Unexpected: {other:?}"),
        }

        drop(client_stream);
        server_thread.join().unwrap();
    }

    #[test]
    fn test_corrupt_and_unsupported_frames_handling() {
        let (server_fd, client_fd) = k3_core::ipc::protocol::create_uds_socket_pair().unwrap();
        let server_stream = UnixStream::from(server_fd);
        let client_stream = UnixStream::from(client_fd);

        let holder = Arc::new(KvHolder::with_memlock_ceiling(1024 * 1024));
        let holder_clone = Arc::clone(&holder);

        let server_thread = std::thread::spawn(move || {
            let _ = handle_connection(&server_stream, &holder_clone);
        });

        // 1. Send compute worker command (ForwardPassCommand, Type ID: 202)
        let worker_cmd = IpcMessage::ForwardPassCommand(k3_core::ipc::protocol::ForwardPassCommand {
            request_id: 1,
            sequence_id: 1,
            token_offset: 0,
            token_count: 1,
            block_indices: vec![0],
            is_prefill: false,
        });
        send_msg_frame(&client_stream, &worker_cmd, None).unwrap();
        let (resp1, _) = recv_msg_frame(&client_stream).unwrap();
        match resp1 {
            IpcMessage::AllocateArenaResponse(r) => {
                assert_eq!(r.status, StatusCode::CapabilityMismatch);
            }
            other => panic!("Unexpected: {other:?}"),
        }

        // 2. Send corrupted Fory payload with known Type ID 101
        let garbage_payload = b"TOTALLY_CORRUPTED_FORY_DATA";
        k3_core::ipc::protocol::send_frame(client_stream.as_raw_fd(), 101, garbage_payload, None).unwrap();
        let (resp2, _) = recv_msg_frame(&client_stream).unwrap();
        match resp2 {
            IpcMessage::AllocateArenaResponse(r) => {
                assert_eq!(r.status, StatusCode::ProtocolError);
            }
            other => panic!("Unexpected: {other:?}"),
        }

        drop(client_stream);
        let _ = server_thread.join();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_stale_socket_cleanup_and_graceful_shutdown() {
        let tmp_dir = std::env::temp_dir().join(format!("k3_kv_srv_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();
        let sock_path = tmp_dir.join("k3-kv.sock");

        // Create a stale dummy file
        fs::write(&sock_path, b"stale data").unwrap();
        assert!(sock_path.exists());

        let config = ServerConfig {
            socket_path: sock_path.clone(),
            max_memory_bytes: 1024 * 1024 * 1024,
        };
        let server = Arc::new(KvServer::new(config));
        let server_clone = Arc::clone(&server);

        let handle = tokio::spawn(async move {
            server_clone.run().await.unwrap();
        });

        // Run client connection & ping in spawn_blocking to avoid blocking async runtime
        let sock_path_client = sock_path.clone();
        tokio::task::spawn_blocking(move || {
            for _ in 0..50 {
                if sock_path_client.exists() {
                    if let Ok(client) = UnixStream::connect(&sock_path_client) {
                        let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 999 });
                        send_msg_frame(&client, &ping, None).unwrap();
                        let (resp, _) = recv_msg_frame(&client).unwrap();
                        assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 999 }));
                        drop(client);
                        return;
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            panic!("failed to connect to socket in 50 attempts");
        })
        .await
        .unwrap();

        // Trigger shutdown
        server.shutdown();
        handle.await.unwrap();

        // Verify socket file was unlinked by SocketFileGuard
        assert!(!sock_path.exists(), "Socket file must be unlinked on shutdown");
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_client_connections() {
        let tmp_dir = std::env::temp_dir().join(format!("k3_kv_con_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();
        let sock_path = tmp_dir.join("k3-kv.sock");

        let config = ServerConfig {
            socket_path: sock_path.clone(),
            max_memory_bytes: 1024 * 1024 * 1024,
        };
        let server = Arc::new(KvServer::new(config));
        let server_clone = Arc::clone(&server);

        let handle = tokio::spawn(async move {
            server_clone.run().await.unwrap();
        });

        // Wait until server is accepting
        for _ in 0..50 {
            if UnixStream::connect(&sock_path).is_ok() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Spawn 6 concurrent clients
        let mut tasks = Vec::new();
        for i in 0..6 {
            let path = sock_path.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                let client = UnixStream::connect(&path).unwrap();
                let req = IpcMessage::AllocateArenaRequest(AllocateArenaRequest {
                    model_id: format!("model-{i}"),
                    arena_bytes: 65536,
                    n_ctx: 64,
                    max_blocks: 4,
                });
                send_msg_frame(&client, &req, None).unwrap();
                let (resp, _) = recv_msg_frame(&client).unwrap();
                let arena_id = match resp {
                    IpcMessage::AllocateArenaResponse(r) => {
                        assert_eq!(r.status, StatusCode::Success);
                        r.arena_id
                    }
                    _ => panic!(),
                };

                let get_fd = IpcMessage::GetArenaFdRequest(GetArenaFdRequest { arena_id });
                send_msg_frame(&client, &get_fd, None).unwrap();
                let (_, opt_fd) = recv_msg_frame(&client).unwrap();
                assert!(opt_fd.is_some());

                let dealloc = IpcMessage::DeallocateArenaRequest(DeallocateArenaRequest { arena_id });
                send_msg_frame(&client, &dealloc, None).unwrap();
                let (resp_d, _) = recv_msg_frame(&client).unwrap();
                match resp_d {
                    IpcMessage::DeallocateArenaResponse(r) => assert_eq!(r.status, StatusCode::Success),
                    _ => panic!(),
                }
                drop(client);
            }));
        }

        for t in tasks {
            t.await.unwrap();
        }

        server.shutdown();
        handle.await.unwrap();
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_server_shutdown_unblocks_connected_client_task() {
        use std::io::Read;

        let tmp_dir = std::env::temp_dir().join(format!("k3_kv_unblock_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();
        let sock_path = tmp_dir.join("k3-kv.sock");

        let config = ServerConfig {
            socket_path: sock_path.clone(),
            max_memory_bytes: 1024 * 1024 * 1024,
        };
        let server = Arc::new(KvServer::new(config));
        let server_clone = Arc::clone(&server);

        let handle = tokio::spawn(async move {
            server_clone.run().await.unwrap();
        });

        // Wait until server is accepting
        let mut client_opt = None;
        for _ in 0..50 {
            if sock_path.exists() {
                if let Ok(client) = UnixStream::connect(&sock_path) {
                    client_opt = Some(client);
                    break;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let client = client_opt.expect("client must connect");

        // Send a ping to verify connection is active and handled by worker thread
        let ping = IpcMessage::PingRequest(PingRequest { timestamp_ns: 777 });
        send_msg_frame(&client, &ping, None).unwrap();
        let (resp, _) = recv_msg_frame(&client).unwrap();
        assert_eq!(resp, IpcMessage::PingResponse(PingResponse { timestamp_ns: 777 }));

        assert_eq!(server.active_client_count(), 1);

        // Keep client open and call server.shutdown()
        server.shutdown();

        // Server run() loop and blocking tasks must terminate promptly without hanging
        tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("server handle must exit within 2s on shutdown")
            .unwrap();

        // Reading from client should now report EOF or error because server shut it down
        let mut buf = [0u8; 16];
        let mut client_ref = &client;
        let read_res = client_ref.read(&mut buf);
        match read_res {
            Ok(0) => {} // EOF
            Err(e) if e.kind() == io::ErrorKind::ConnectionReset => {}
            Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {}
            Err(e) if e.kind() == io::ErrorKind::NotConnected => {}
            other => panic!("Expected EOF or connection error, got {other:?}"),
        }

        drop(client);
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
