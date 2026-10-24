//! Context memory holder daemon and shared memory arena management for the `SpacemiT` K3 AI platform.

pub const KV_DAEMON_NAME: &str = "k3-kv";

pub mod holder;
pub mod server;

pub use holder::{
    ArenaAllocationConfig, ArenaAllocationResult, ComputedArenaLayout, KvError, KvHolder,
    compute_arena_layout,
};
pub use server::{
    DEFAULT_SOCKET_PATH, KvServer, ServerConfig, SocketFileGuard, handle_connection,
    prepare_and_bind_socket,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_identity() {
        assert_eq!(KV_DAEMON_NAME, "k3-kv");
    }
}
