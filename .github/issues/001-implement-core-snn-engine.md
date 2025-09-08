Title: Implement core SNN engine

Description:
Implement the core Spiking Neural Network (SNN) engine with a focus on high performance and correctness. This includes an event-driven simulation loop, support for the Leaky Integrate-and-Fire (LIF) neuron model (and optional Izhikevich), sparse updates, and surrogate gradient support for training.

Acceptance criteria:
- LIF neuron implementation with configurable parameters (tau, threshold, reset, refractory).
- Event-driven simulation loop capable of processing spike events via an efficient event queue.
- Support for injecting external current and wiring synaptic connections.
- Unit tests for LIF dynamics (happy path + spike/reset + refractory).
- Benchmarks showing basic throughput for small networks.

Labels: feature, core, snn
Estimated effort: 5d
Assignee: 
Status: open
