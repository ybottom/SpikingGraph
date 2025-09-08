A general-purpose **S**piking **N**eural **N**etwork library, with the latest training and inference techniques and optimisations. 

# Goals

The primary intention is for this library to be used to create, train and execute SNNs efficiently and intelligently on a variety of consumer grade hardware. with a focus on integration with modern game engines like Unity, Godot, and Unreal. Its primary design principle prioritizes speed and efficiency within resource-constrained game engine environments.
## Features
- Highest possible performance and efficiency
- Runs on any consumer grade hardware: 
  - CPU,
  - NPU,
  - and GPU.
- Can leverage any available consumer hardware to work in tandem and in parallel / distribute work efficiently
- Real time inference with minimal resource usage
- Full language bindings for:
  - C,
  - C++,
  - Python,
  - Rust,
  - C# / .net 9,

## Implementation Strategy

### Modern SNN Training and Inference Techniques

- **Surrogate Gradient Descent**: Use efficient surrogate gradient methods for training pure SNNs. These approaches allow the use of familiar backpropagation algorithms by approximating the non-differentiable spiking function with a differentiable surrogate in both forward and backward passes, optimizing for performance on standard consumer GPUs.[^1][^2]
- **Direct SNN Architectures**: Prioritize direct SNN designs (not ANN-to-SNN conversions). Use neuron models like Leaky Integrate-and-Fire (LIF), adaptive LIF, or Izhikevich neurons, implemented with optimized tensor operations that map efficiently to CUDA and OpenCL kernels.[^3][^1]
- **Event-Driven Simulation**: Employ event-driven computation and sparse updates. Process only active neurons or synapses each timestep rather than dense tensor operations, maximizing memory and compute efficiency on consumer GPUs.[^1]


### Consumer Hardware Optimization

#### GPU (CUDA/OpenCL)

- **Pre-Built Kernels**: Develop hand-optimized CUDA and OpenCL kernels for:
    - Neuron membrane potential updates
    - Spike propagation and accumulation
    - Synaptic weight updates and plasticity rules (e.g., STDP)
- **Batching and Streaming**: Support batch-mode SNN inference for high-throughput pipelines as well as low-latency streaming for real-time use cases—using pinned memory and zero-copy buffers for direct GPU ↔ host access.[^4][^1]
- **Tensor Mapping**: Use contiguous tensors, avoid host-device memory copies with unified memory, and expose direct device context handles for advanced users.


#### CPU (x86, AMD64)

- **SIMD**: Implement SIMD vectorization for core update steps, leveraging AVX2/AVX512 on modern Intel/AMD CPUs.
- **Thread Pool**: Create thread pools for multi-core parallelism and lockless queues for spike event propagation.
- **Memory Layouts**: Use cache-friendly data layouts (SoA vs AoS) for neuron and synapse arrays.


#### NPU (Neural Processing Unit)

- **API Hooks**: Provide explicit entry points for NPUs if available (e.g., on Intel/Amd consumer chips), using vendor-specific acceleration libraries if supported.[^5][^6]
- **Fallback Path**: If no NPU is present, gracefully fallback to CPU/GPU execution paths.


### DLL/API and Language Interoperability

- **C/C++ Core DLL**: Implement the core SNN logic in C or C++ for maximum portability, exposing functions through a clean, versioned API (using `extern "C"` for C compatibility).
- **FFI Bindings**: Provide Foreign Function Interface bindings for Python (ctypes, cffi), Rust, Java (JNI), C\#, etc., using standardized calling conventions (cdecl, stdcall) and handle types (opaque pointers for network/context).
- **In-memory Data Structures**:
    - Support direct buffer sharing (pointers, mapped arrays) to allow zero-copy data interchange.
    - Expose reference counting or explicit allocation/deallocation methods for cross-language memory management.
- **Thread Safety**: Mark entry points as thread-safe where possible; provide context handles for concurrent model execution.
- **Callback Hooks**: Enable user-defined callbacks for spike events, synaptic updates, or end-of-epoch signals, allowing fast interop and integration for real-time systems.[^7][^8]


### Example API Design

```cpp
// C API example for DLL entry points
typedef void* SNNHandle;
typedef void (*SpikeCallback)(int neuron_id, double time, void* user_data);

SNNHandle snn_create_network(const SNNConfig* config);
int snn_load_weights(SNNHandle snn, const char* filename);
int snn_run_step(SNNHandle snn, double dt);
int snn_register_spike_callback(SNNHandle snn, SpikeCallback cb, void* user_data);
int snn_get_membrane_potentials(SNNHandle snn, float* buffer, int length);
int snn_destroy_network(SNNHandle snn);
```

Bindings to other languages are straightforward and allow in-memory, real-time communication.

## Core Library Components

- **Core SNN Engine**: Real-time inference/training loop using direct memory access and optimized compute kernels.[^3][^1]
- **Interoperability Layer**: Direct API for buffer sharing, callbacks, and FFI bindings.
- **GPU/CPU/NPU Backends**: Auto-detection and dynamic allocation to the best available device.
- **Developer Hooks**: Callbacks, event listeners, and memory sharing for seamless integration.
- **Documentation**: Explicit instructions for linking, memory management, calling conventions, and multi-language usage.


## Performance Considerations

- With careful kernel and threading design, pure SNN architectures achieve sub-millisecond inference latencies and can scale up to millions of neurons/synapses on modern RTX/AMD GPUs.
- Direct in-memory APIs and buffer sharing eliminate latency and allow maximal throughput for demanding applications (robotics, AR/VR, scientific simulation).


## References

- Modern SNN GPU/CPU optimizations[^4][^1][^3]
- API/DLL-oriented SNN libraries[^8][^7]
- Consumer hardware benchmarks[^9][^10][^11]
- Interoperability design[^6][^5]

This approach ensures real-time, high-performance SNN computation with seamless integration into any software ecosystem on **entirely consumer hardware**—no servers, endpoints, or exotic silicon required.

<div style="text-align: center">⁂</div>

[^1]: https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full

[^2]: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full

[^3]: https://ics.uci.edu/~jmoorkan/pub/gpusnn-ijcnn.pdf

[^4]: https://www.nature.com/articles/s41598-019-54957-7

[^5]: https://www.forbes.com/sites/moorinsights/2024/04/29/at-the-heart-of-the-ai-pc-battle-lies-the-npu/

[^6]: https://www.aiacceleratorinstitute.com/improving-ai-inference-performance-with-hardware-accelerators/

[^7]: https://github.com/michaelmelanson/spiking-neural-net

[^8]: https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00089/full

[^9]: https://lambda.ai/blog/nvidia-rtx-4090-vs-rtx-3090-deep-learning-benchmark

[^10]: https://semianalysis.com/2025/05/23/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens/

[^11]: https://www.tomshardware.com/reviews/cpu-hierarchy,4312.html
