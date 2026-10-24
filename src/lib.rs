//! Context memory holder daemon and shared memory arena management for the `SpacemiT` K3 AI platform.

pub const KV_DAEMON_NAME: &str = "k3-kv";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_identity() {
        assert_eq!(KV_DAEMON_NAME, "k3-kv");
    }
}
