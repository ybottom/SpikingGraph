# Core Architecture and Components

## Overview
SpikingGraph is a high-performance, general-purpose Spiking Neural Network (SNN) library designed for real-time training and inference on consumer-grade hardware. The architecture is modular, hardware-agnostic, and optimized for integration with modern game engines and other real-time systems.

## Core Components

### 1. Core SNN Engine
- Implements real-time inference and training loops.
- Uses event-driven simulation and sparse updates for efficiency.
- Supports modern neuron models (LIF, adaptive LIF, Izhikevich).
- Surrogate gradient descent for training.
- Optimized for direct memory access and minimal latency.

### 2. Hardware Backends
- **GPU Backend:** Hand-optimized CUDA/OpenCL kernels for neuron updates, spike propagation, and synaptic plasticity. Supports batching, streaming, and zero-copy buffers.
- **CPU Backend:** SIMD vectorization, thread pools, and cache-friendly memory layouts for efficient parallelism.
- **NPU Backend:** API hooks for vendor-specific acceleration; automatic fallback to CPU/GPU if unavailable.
- Auto-detection and dynamic allocation to the best available device.

### 3. Interoperability Layer
- C/C++ core DLL with a clean, versioned API.
- FFI bindings for Python, Rust, C#, Java, etc.
- Direct buffer sharing and zero-copy data interchange.
- Thread-safe entry points and context handles for concurrent execution.
- Callback hooks for spike events, synaptic updates, and custom integration.

### 4. Developer Hooks and Extensibility
- User-defined callbacks for real-time event handling.
- Event listeners and memory sharing for seamless integration with external systems.
- Explicit allocation/deallocation and reference counting for cross-language memory management.

### 5. Documentation and API Reference
- Clear instructions for linking, memory management, and calling conventions.
- Multi-language usage examples and integration guides.

## Data Flow
1. **Initialization:** User configures and creates an SNN network via the API.
2. **Execution:** The engine runs inference/training steps, leveraging the optimal hardware backend.
3. **Interoperability:** Data and events are shared in real-time with external systems via FFI and callbacks.
4. **Resource Management:** Explicit APIs for allocation, deallocation, and buffer sharing ensure efficient memory usage.

## Example API (C)
```cpp
SNNHandle snn_create_network(const SNNConfig* config);
int snn_load_weights(SNNHandle snn, const char* filename);
int snn_run_step(SNNHandle snn, double dt);
int snn_register_spike_callback(SNNHandle snn, SpikeCallback cb, void* user_data);
int snn_get_membrane_potentials(SNNHandle snn, float* buffer, int length);
int snn_destroy_network(SNNHandle snn);
```

---

This architecture ensures maximal performance, flexibility, and ease of integration for any real-time SNN application on consumer hardware.
