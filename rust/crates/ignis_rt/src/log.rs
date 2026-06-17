//! Minimal file logger. The addon calls `ignis_set_log_path()` (-> user home) and
//! `ignis_set_base_path()` (-> the addon folder). We log to BOTH: the primary path, and a
//! mirror inside the addon folder (`<base>/ignis-rs.log`) which is reachable over the SMB
//! share used for deploys — so the renderer's bring-up can be inspected remotely.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

static LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
static MIRROR_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Set the primary log file path and write a start banner.
pub fn set_path(path: &str) {
    *LOG_PATH.lock().unwrap() = Some(PathBuf::from(path));
    log("=== ignis_rt (Rust) log start ===");
}

/// Add a mirror log next to the DLL/addon (truncated fresh each session).
pub fn set_mirror(path: &str) {
    let p = PathBuf::from(path);
    let _ = std::fs::write(&p, b"=== ignis_rt (Rust) mirror log ===\n");
    *MIRROR_PATH.lock().unwrap() = Some(p);
}

/// Append one line to the log file(s) and stderr.
pub fn log(msg: &str) {
    for slot in [&LOG_PATH, &MIRROR_PATH] {
        if let Some(path) = slot.lock().unwrap().as_ref() {
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "[ignis-rs] {msg}");
            }
        }
    }
    eprintln!("[ignis-rs] {msg}");
}
