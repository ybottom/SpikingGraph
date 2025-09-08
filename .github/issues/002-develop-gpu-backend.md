Title: Develop GPU backend (CUDA/OpenCL)

Description:
Implement a GPU backend for inference and training. Provide kernels for spike propagation and neuron updates that can process batched spikes efficiently. Support SELL/CSR conversions and zero-copy buffers where feasible.

Acceptance criteria:
- CUDA kernels for spike propagation and neuron update.
- Data structures to convert graph into SELL format for GPU inference.
- Example demonstrating GPU inference performance vs CPU.
- Fallback path to CPU when GPU unavailable.

Labels: feature, backend, gpu
Estimated effort: 10d
Assignee:
Status: open
