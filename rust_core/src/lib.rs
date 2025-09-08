use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_float, c_int, c_void};
use std::sync::{Arc, Mutex};
use std::ptr;

#[repr(C)]
pub struct SNNConfig {
    pub num_neurons: c_int,
    pub dt: c_double,
}

type SpikeCallback = Option<extern "C" fn(neuron_id: c_int, time: c_double, user_data: *mut c_void)>;

struct SNNContext {
    num_neurons: usize,
    dt: f64,
    membrane_potentials: Vec<f32>,
    thresholds: Vec<f32>,
    resets: Vec<f32>,
    taus: Vec<f32>,
    connections: Vec<Vec<(usize, f32)>>,
    input_current: Vec<f32>,
    callback: SpikeCallback,
    user_data: *mut c_void,
}

#[no_mangle]
pub extern "C" fn snn_create_network(cfg: *const SNNConfig) -> *mut c_void {
    if cfg.is_null() { return ptr::null_mut(); }
    let cfg = unsafe { &*cfg };
    let n = cfg.num_neurons as usize;
    let ctx = SNNContext {
        num_neurons: n,
        dt: cfg.dt as f64,
        membrane_potentials: vec![0.0; n],
        thresholds: vec![1.0; n],
        resets: vec![0.0; n],
        taus: vec![10.0; n],
        connections: vec![Vec::new(); n],
        input_current: vec![0.0; n],
        callback: None,
        user_data: ptr::null_mut(),
    };
    Box::into_raw(Box::new(Mutex::new(ctx))) as *mut c_void
}

#[no_mangle]
pub extern "C" fn snn_destroy_network(handle: *mut c_void) -> c_int {
    if handle.is_null() { return -1; }
    unsafe { Box::from_raw(handle as *mut Mutex<SNNContext>); }
    0
}

#[no_mangle]
pub extern "C" fn snn_run_step(handle: *mut c_void, dt: c_double) -> c_int {
    if handle.is_null() { return -1; }
    let mutex = unsafe { &*(handle as *mut Mutex<SNNContext>) };
    let mut ctx = mutex.lock().unwrap();
    for i in 0..ctx.num_neurons {
        let I = ctx.input_current[i];
        let V = ctx.membrane_potentials[i];
        let tau = ctx.taus[i];
        let dV = (-V / tau + I) * dt as f32;
        let mut Vn = V + dV;
        ctx.input_current[i] = 0.0;
        if Vn >= ctx.thresholds[i] {
            Vn = ctx.resets[i];
            if let Some(cb) = ctx.callback {
                cb(i as c_int, 0.0, ctx.user_data);
            }
            for (dst, w) in ctx.connections[i].iter() {
                if *dst < ctx.num_neurons {
                    ctx.input_current[*dst] += *w;
                }
            }
        }
        ctx.membrane_potentials[i] = Vn;
    }
    0
}

#[no_mangle]
pub extern "C" fn snn_inject_current(handle: *mut c_void, neuron_id: c_int, current: c_float) -> c_int {
    if handle.is_null() { return -1; }
    let mutex = unsafe { &*(handle as *mut Mutex<SNNContext>) };
    let mut ctx = mutex.lock().unwrap();
    let id = neuron_id as usize;
    if id >= ctx.num_neurons { return -1; }
    ctx.input_current[id] += current;
    0
}

#[no_mangle]
pub extern "C" fn snn_add_connection(handle: *mut c_void, src: c_int, dst: c_int, weight: c_float) -> c_int {
    if handle.is_null() { return -1; }
    let mutex = unsafe { &*(handle as *mut Mutex<SNNContext>) };
    let mut ctx = mutex.lock().unwrap();
    let s = src as usize; let d = dst as usize;
    if s >= ctx.num_neurons || d >= ctx.num_neurons { return -1; }
    ctx.connections[s].push((d, weight));
    0
}

#[no_mangle]
pub extern "C" fn snn_register_spike_callback(handle: *mut c_void, cb: SpikeCallback, user_data: *mut c_void) -> c_int {
    if handle.is_null() { return -1; }
    let mutex = unsafe { &*(handle as *mut Mutex<SNNContext>) };
    let mut ctx = mutex.lock().unwrap();
    ctx.callback = cb;
    ctx.user_data = user_data;
    0
}

#[no_mangle]
pub extern "C" fn snn_get_membrane_potentials(handle: *mut c_void, buffer: *mut c_float, length: c_int) -> c_int {
    if handle.is_null() || buffer.is_null() { return -1; }
    let mutex = unsafe { &*(handle as *mut Mutex<SNNContext>) };
    let ctx = mutex.lock().unwrap();
    let to_copy = std::cmp::min(length as usize, ctx.num_neurons);
    unsafe {
        for i in 0..to_copy {
            *buffer.add(i) = ctx.membrane_potentials[i];
        }
    }
    to_copy as c_int
}
