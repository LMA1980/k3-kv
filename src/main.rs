//! Standalone daemon binary for `k3-kv`: Context memory holder for the `SpacemiT` K3 platform.
//!
//! Enforces operator memory lock bounds (ADR-037), holds isolated per-model shared-memory
//! arenas (ADR-042, ADR-043), and passes arena file descriptors via `SCM_RIGHTS` over UDS.

use std::env;
use std::fs;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::process::ExitCode;

use k3_core::ipc::shm::DEFAULT_LIMIT_MEMLOCK_BYTES;
use k3_kv::holder::KvHolder;
use k3_kv::server::{KvServer, SocketFileGuard};

/// Command line configuration for `k3-kv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvConfig {
    /// Path to the Unix Domain Socket listener.
    pub socket_path: PathBuf,
    /// Maximum lockable memory bytes for the holder (operator ceiling).
    pub max_memory_bytes: u64,
    /// Verbose diagnostic output flag.
    pub verbose: bool,
}

impl Default for KvConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from("/tmp/k3-kv.sock"),
            max_memory_bytes: DEFAULT_LIMIT_MEMLOCK_BYTES, // 28 GiB
            verbose: false,
        }
    }
}

impl KvConfig {
    /// Parses CLI arguments from an iterator of strings.
    ///
    /// # Errors
    /// Returns `Err` with a diagnostic message if arguments are malformed or missing required values.
    pub fn parse_args<I>(mut args: I) -> Result<Option<Self>, String>
    where
        I: Iterator<Item = String>,
    {
        let mut config = Self::default();

        // Skip binary name if present
        let _ = args.next();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    return Ok(None);
                }
                "--version" | "-V" => {
                    println!("k3-kv {}", env!("CARGO_PKG_VERSION"));
                    return Ok(None);
                }
                "--socket" => {
                    let path = args
                        .next()
                        .ok_or_else(|| "--socket requires a file path".to_owned())?;
                    if path.trim().is_empty() {
                        return Err("--socket requires a non-empty file path".to_owned());
                    }
                    config.socket_path = PathBuf::from(path);
                }
                "--max-memory-bytes" => {
                    let val = args
                        .next()
                        .ok_or_else(|| "--max-memory-bytes requires a byte count".to_owned())?;
                    let bytes = parse_memory_bytes(&val)
                        .map_err(|e| format!("invalid --max-memory-bytes '{val}': {e}"))?;
                    if bytes == 0 {
                        return Err("--max-memory-bytes must be greater than 0".to_owned());
                    }
                    config.max_memory_bytes = bytes;
                }
                "--verbose" => {
                    config.verbose = true;
                }
                unknown => {
                    return Err(format!("unknown option '{unknown}'"));
                }
            }
        }

        Ok(Some(config))
    }
}

/// Parses byte strings supporting raw integers or unit suffixes (`GiB`, `GB`, `MiB`, `MB`, etc.).
fn parse_memory_bytes(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty byte string".to_owned());
    }

    if let Ok(bytes) = s.parse::<u64>() {
        return Ok(bytes);
    }

    let s_upper = s.to_ascii_uppercase();
    let (num_str, multiplier) = if let Some(n) = s_upper.strip_suffix("GIB") {
        (n, 1024 * 1024 * 1024_u64)
    } else if let Some(n) = s_upper.strip_suffix("GB") {
        (n, 1_000_000_000_u64)
    } else if let Some(n) = s_upper.strip_suffix('G') {
        (n, 1024 * 1024 * 1024_u64)
    } else if let Some(n) = s_upper.strip_suffix("MIB") {
        (n, 1024 * 1024_u64)
    } else if let Some(n) = s_upper.strip_suffix("MB") {
        (n, 1_000_000_u64)
    } else if let Some(n) = s_upper.strip_suffix('M') {
        (n, 1024 * 1024_u64)
    } else if let Some(n) = s_upper.strip_suffix("KIB") {
        (n, 1024_u64)
    } else if let Some(n) = s_upper.strip_suffix("KB") {
        (n, 1000_u64)
    } else if let Some(n) = s_upper.strip_suffix('K') {
        (n, 1024_u64)
    } else {
        return Err("expected integer or suffix (e.g. 28GiB, 1024MiB)".to_owned());
    };

    let num = num_str
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("cannot parse number '{num_str}'"))?;
    num.checked_mul(multiplier)
        .ok_or_else(|| "byte calculation overflowed u64".to_owned())
}

fn print_usage() {
    println!(
        r"k3-kv: Context memory holder daemon for the SpacemiT K3 platform.

USAGE:
    k3-kv [OPTIONS]

OPTIONS:
    --socket <path>              Path to Unix Domain Socket [default: /tmp/k3-kv.sock]
    --max-memory-bytes <bytes>   Maximum memory ceiling in bytes [default: 28 GiB]
    --verbose                    Enable verbose diagnostic output
    -h, --help                   Print help information
    -V, --version                Print version information

EXIT CODES:
    0    Graceful shutdown or successful query (--help, --version)
    1    Runtime error (socket binding failure, active daemon conflict)
    2    Invalid command line arguments
"
    );
}

#[allow(clippy::too_many_lines)]
#[tokio::main]
async fn main() -> ExitCode {
    let args = env::args();
    let config = match KvConfig::parse_args(args) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => return ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("[error] {err}");
            eprintln!("Run 'k3-kv --help' for usage instructions.");
            return ExitCode::from(2);
        }
    };

    if config.verbose {
        eprintln!(
            "[k3-kv] starting daemon (socket: {}, max_memory: {} bytes / {} GiB)",
            config.socket_path.display(),
            config.max_memory_bytes,
            config.max_memory_bytes / (1024 * 1024 * 1024)
        );
    }

    // Register OS termination signal handlers
    let mut sigint = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[error] failed to register SIGINT handler: {e}");
            return ExitCode::from(1);
        }
    };

    let mut sigterm = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[error] failed to register SIGTERM handler: {e}");
            return ExitCode::from(1);
        }
    };

    // Stale socket detection and clean startup
    if config.socket_path.exists() {
        if UnixStream::connect(&config.socket_path).is_ok() {
            eprintln!(
                "[error] another k3-kv daemon is already running on {}",
                config.socket_path.display()
            );
            return ExitCode::from(1);
        }
        if config.verbose {
            eprintln!(
                "[k3-kv] removing stale socket: {}",
                config.socket_path.display()
            );
        }
        let _ = fs::remove_file(&config.socket_path);
    }

    // Ensure parent directory exists
    if let Some(parent) = config.socket_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("[error] failed to create socket parent dir {}: {e}", parent.display());
                return ExitCode::from(1);
            }
        }
    }

    let _socket_guard = SocketFileGuard::new(config.socket_path.clone());

    let holder = KvHolder::with_memlock_ceiling(config.max_memory_bytes);
    let server = match KvServer::bind(&config.socket_path, holder) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[error] failed to bind socket {}: {e}",
                config.socket_path.display()
            );
            return ExitCode::from(1);
        }
    };

    let shutdown_handle = server.shutdown_handle();
    let server_task = tokio::spawn(async move {
        server.run().await
    });

    if config.verbose {
        eprintln!(
            "[k3-kv] listener ready on {}",
            config.socket_path.display()
        );
    }

    // Await server completion or OS termination signal
    tokio::select! {
        _ = sigint.recv() => {
            if config.verbose {
                eprintln!("[k3-kv] received SIGINT (Ctrl+C), initiating graceful shutdown...");
            }
            shutdown_handle.notify_waiters();
            shutdown_handle.notify_one();
        }
        _ = sigterm.recv() => {
            if config.verbose {
                eprintln!("[k3-kv] received SIGTERM, initiating graceful shutdown...");
            }
            shutdown_handle.notify_waiters();
            shutdown_handle.notify_one();
        }
        res = server_task => {
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    eprintln!("[error] server loop encountered error: {e}");
                    return ExitCode::from(1);
                }
                Err(e) => {
                    eprintln!("[error] server task panicked: {e}");
                    return ExitCode::from(1);
                }
            }
            return ExitCode::SUCCESS;
        }
    }

    if config.verbose {
        eprintln!("[k3-kv] shutdown completed, unlinking socket");
    }

    ExitCode::SUCCESS
}
