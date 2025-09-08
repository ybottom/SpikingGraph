use std::collections::BinaryHeap;
use std::cmp::Ordering;
#[derive(Eq, PartialEq)]
struct Event {
    time: f64,
    neuron_id: usize,
    weight: f32,
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.time.partial_cmp(&self.time).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_float, c_int, c_void};
use std::sync::{Arc, Mutex};
use std::ptr;

#[repr(C)]
pub enum NeuronModelType {
    LIF = 0,
    Izhikevich = 1,
}

#[repr(C)]
pub struct SNNConfig {
    pub num_neurons: c_int,
    pub dt: c_double,
    pub neuron_model: c_int, // 0 = LIF, 1 = Izhikevich
}

type SpikeCallback = Option<extern "C" fn(neuron_id: c_int, time: c_double, user_data: *mut c_void)>;

struct SNNContext {
    num_neurons: usize,
    dt: f64,
    neuron_model: NeuronModelType,
    // LIF state
    membrane_potentials: Vec<f32>,
    thresholds: Vec<f32>,
    resets: Vec<f32>,
    taus: Vec<f32>,
    // Izhikevich state
    izh_v: Vec<f32>,
    izh_u: Vec<f32>,
    izh_a: Vec<f32>,
    izh_b: Vec<f32>,
    izh_c: Vec<f32>,
    izh_d: Vec<f32>,
    // Common
    connections: Vec<Vec<(usize, f32)>>,
    input_current: Vec<f32>,
    callback: SpikeCallback,
    user_data: *mut c_void,
    // Event-driven
    event_queue: BinaryHeap<Event>,
    sim_time: f64,
    synaptic_delay: f64, // fixed delay for all synapses for now
}

#[no_mangle]
pub extern "C" fn snn_create_network(cfg: *const SNNConfig) -> *mut c_void {
    if cfg.is_null() { return ptr::null_mut(); }
    let cfg = unsafe { &*cfg };
    let n = cfg.num_neurons as usize;
    let neuron_model = match cfg.neuron_model {
        1 => NeuronModelType::Izhikevich,
        _ => NeuronModelType::LIF,
    };
    let ctx = SNNContext {
        num_neurons: n,
        dt: cfg.dt as f64,
        neuron_model,
        // LIF
        membrane_potentials: vec![0.0; n],
        thresholds: vec![1.0; n],
        resets: vec![0.0; n],
        taus: vec![10.0; n],
        // Izhikevich
        izh_v: vec![-65.0; n],
        izh_u: vec![0.0; n],
        izh_a: vec![0.02; n],
        izh_b: vec![0.2; n],
        izh_c: vec![-65.0; n],
        izh_d: vec![8.0; n],
    // Common
    connections: vec![Vec::new(); n],
    input_current: vec![0.0; n],
    callback: None,
    user_data: ptr::null_mut(),
    // Event-driven
    event_queue: BinaryHeap::new(),
    sim_time: 0.0,
    synaptic_delay: 1.0, // ms, can be made configurable
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
    let end_time = ctx.sim_time + dt as f64;
    // Process all events up to end_time
    while let Some(ev) = ctx.event_queue.peek() {
        if ev.time > end_time { break; }
        let ev = ctx.event_queue.pop().unwrap();
        ctx.sim_time = ev.time;
        let i = ev.neuron_id;
        // Deliver input
        ctx.input_current[i] += ev.weight;
        // Update neuron state and check for spike
        match ctx.neuron_model {
            NeuronModelType::LIF => {
                let V = ctx.membrane_potentials[i];
                let tau = ctx.taus[i];
                let I = ctx.input_current[i];
                let dV = (-V / tau + I) * (ctx.sim_time as f32 - ctx.sim_time as f32); // dt is 0 for event, so use input as impulse
                let mut Vn = V + dV + I; // treat event as instantaneous input
                ctx.input_current[i] = 0.0;
                if Vn >= ctx.thresholds[i] {
                    Vn = ctx.resets[i];
                    if let Some(cb) = ctx.callback {
                        cb(i as c_int, ctx.sim_time, ctx.user_data);
                    }
                    for (dst, w) in ctx.connections[i].iter() {
                        if *dst < ctx.num_neurons {
                            ctx.event_queue.push(Event {
                                time: ctx.sim_time + ctx.synaptic_delay,
                                neuron_id: *dst,
                                weight: *w,
                            });
                        }
                    }
                }
                ctx.membrane_potentials[i] = Vn;
            }
            NeuronModelType::Izhikevich => {
                let v = ctx.izh_v[i];
                let u = ctx.izh_u[i];
                let a = ctx.izh_a[i];
                let b = ctx.izh_b[i];
                let c = ctx.izh_c[i];
                let d = ctx.izh_d[i];
                let I = ctx.input_current[i];
                let dv = (0.04 * v * v + 5.0 * v + 140.0 - u + I);
                let du = (a * (b * v - u));
                let mut vn = v + dv;
                let mut un = u + du;
                ctx.input_current[i] = 0.0;
                if vn >= 30.0 {
                    vn = c;
                    un += d;
                    if let Some(cb) = ctx.callback {
                        cb(i as c_int, ctx.sim_time, ctx.user_data);
                    }
                    for (dst, w) in ctx.connections[i].iter() {
                        if *dst < ctx.num_neurons {
                            ctx.event_queue.push(Event {
                                time: ctx.sim_time + ctx.synaptic_delay,
                                neuron_id: *dst,
                                weight: *w,
                            });
                        }
                    }
                }
                ctx.izh_v[i] = vn;
                ctx.izh_u[i] = un;
            }
        }
    }
    ctx.sim_time = end_time;
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
