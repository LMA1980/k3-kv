//! Context memory holder daemon and shared memory arena management for the `SpacemiT` K3 AI platform.

pub const KV_DAEMON_NAME: &str = "k3-kv";

pub mod holder;
pub mod server;

pub use holder::{
    compute_arena_layout, ArenaAllocationConfig, ArenaAllocationResult, ComputedArenaLayout,
    KvError, KvHolder,
};
pub use server::{
    handle_connection, prepare_and_bind_socket, KvServer, ServerConfig, SocketFileGuard,
    DEFAULT_SOCKET_PATH,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_identity() {
        assert_eq!(KV_DAEMON_NAME, "k3-kv");
    }
}
