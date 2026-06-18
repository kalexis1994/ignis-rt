//! Global key/value config + assorted CPU-side state the addon pushes before/around
//! init (shader_mode, dlss_enabled, spp, the baked tonemap LUT, base path...). Stored
//! now, consumed by the renderer as those subsystems come online.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

struct State {
    ints: HashMap<String, i32>,
    floats: HashMap<String, f32>,
    base_path: String,
    /// (lut_size, rgb data) from ignis_upload_lut — uploaded to the GPU later.
    lut: Option<(u32, Vec<f32>)>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        ints: HashMap::new(),
        floats: HashMap::new(),
        base_path: String::new(),
        lut: None,
    })
});

pub fn set_int(key: &str, value: i32) {
    STATE.lock().unwrap().ints.insert(key.to_string(), value);
}
pub fn get_int(key: &str) -> i32 {
    STATE.lock().unwrap().ints.get(key).copied().unwrap_or(0)
}
pub fn set_float(key: &str, value: f32) {
    STATE.lock().unwrap().floats.insert(key.to_string(), value);
}
pub fn get_float(key: &str) -> f32 {
    STATE.lock().unwrap().floats.get(key).copied().unwrap_or(0.0)
}
pub fn set_base_path(path: &str) {
    STATE.lock().unwrap().base_path = path.to_string();
}
pub fn base_path() -> String {
    STATE.lock().unwrap().base_path.clone()
}
pub fn store_lut(size: u32, data: Vec<f32>) {
    STATE.lock().unwrap().lut = Some((size, data));
}

pub fn get_lut() -> Option<(u32, Vec<f32>)> {
    STATE.lock().unwrap().lut.clone()
}
