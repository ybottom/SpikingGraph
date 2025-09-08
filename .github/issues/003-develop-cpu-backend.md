Title: Develop CPU backend (SIMD, thread pool)

Description:
Implement a high-performance CPU backend optimized for x86_64 using SIMD (AVX2/AVX512), thread pools, and cache-friendly data layouts (SoA). Provide a deterministic event-driven/hybrid simulation loop.

Acceptance criteria:
- SIMD-accelerated neuron updates for LIF layers.
- Thread pool for multi-core parallelism and safe spike propagation.
- Unit tests and microbenchmarks showing scaling across cores.

Labels: feature, backend, cpu
Estimated effort: 6d
Assignee:
Status: open
