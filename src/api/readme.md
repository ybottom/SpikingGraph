

# **A Practical Guide to Implementing a High-Performance Spiking Neural Network Engine**

## **Section 1: Foundational Principles of Spiking Neural Networks**

### **1.1 The Bio-Inspired Paradigm: Spikes, Sparsity, and Event-Driven Computation**

Spiking Neural Networks (SNNs) represent a significant departure from traditional Artificial Neural Networks (ANNs), drawing inspiration directly from the computational principles of the biological brain.1 Their architecture and operation are built upon a triad of interconnected concepts: the use of discrete spikes for information transmission, the resulting sparsity of network activity, and the enablement of an event-driven computational model. This bio-inspired paradigm is not merely a matter of biological mimicry; it is the source of the profound efficiency advantages that SNNs promise, particularly when deployed on specialized neuromorphic hardware.3

The fundamental unit of information in an SNN is the **spike**, a discrete, all-or-nothing event in time. In biology, this is an action potential, an electrical impulse of approximately 100 mV.3 For computational purposes, this complex analog signal is abstracted into a single, binary bit: a '1' representing the occurrence of a spike at a specific time step, and a '0' representing silence.3 This simplification is a cornerstone of SNN efficiency, as representing and transmitting single bits is far less costly in hardware than handling high-precision floating-point values common in ANNs.3

This binary, event-based communication naturally gives rise to **sparsity**. Unlike in many ANNs where most neurons are active and computing at every step, biological neurons spend the majority of their time at rest, firing only when their input stimulus crosses a certain threshold.2 This means that at any given moment, most activations in a large SNN are zero. Sparsity has direct and significant benefits for computational efficiency. Sparse vectors and tensors require less memory to store, and, critically, computations involving multiplication by zero can be entirely skipped, drastically reducing the number of required operations.3

Sparsity is the direct enabler of the **event-driven processing** model, the most significant departure from the clock-based operation of conventional computers and ANNs.6 Because most neurons are inactive, there is no need to update the entire network state at every discrete time step. Instead, computation is only performed when and where an event—a spike—occurs.1 This principle, sometimes referred to as "static suppression," mirrors biological sensory systems that primarily process changes in their environment, filtering out static, unchanging information.3 This event-driven nature is the primary source of the ultra-low power consumption of SNNs on neuromorphic hardware.6

Finally, SNNs are inherently temporal processors. Information is encoded not just in *whether* a neuron spikes, but precisely *when* it spikes.1 This introduces a rich temporal dimension to the computation, allowing SNNs to naturally process time-varying data and recognize spatiotemporal patterns in a way that is less natural for static ANNs.1

### **1.2 Core Components: Neuron Models and Synaptic Dynamics**

The fundamental components of an SNN graph are its nodes (neurons) and its edges (synapses). A spiking neuron can be conceptualized as a dynamic system that integrates inputs over time.2 Its state is primarily defined by its

**membrane potential**, an internal variable that represents the accumulated electrical charge from incoming spikes.1 When this potential crosses a specific

**firing threshold**, the neuron "fires," emitting a spike of its own to all connected downstream neurons.2 Immediately after firing, the neuron enters a

**refractory period** during which its potential is reset and it is temporarily unable to fire again, a mechanism that helps regulate network activity.2

The connections between these neurons are called **synapses**, and their strength is defined by a **synaptic weight**. This weight determines how much a spike from a presynaptic (sending) neuron will influence the membrane potential of a postsynaptic (receiving) neuron.1 Crucially, these weights are not fixed. The ability of synaptic strengths to change over time based on the network's activity is known as

**synaptic plasticity**, and it is the primary mechanism for learning and memory formation in SNNs.1

### **1.3 Information Encoding in Spikes: From Rate to Latency**

Before an SNN can process external data, such as an image or an audio signal, that data must be converted into the native language of the network: spike trains. This conversion process, known as spike encoding, is a critical design choice that fundamentally shapes the network's behavior and efficiency. The method of encoding dictates what temporal features the network will be sensitive to and has a direct impact on the required simulation time and the nature of the learning problem. A library's API must therefore treat encoding as a first-class citizen, offering flexible and configurable modules for this task, as exemplified by the snntorch.spikegen module.3

The primary encoding strategies include:

* **Rate Coding:** In this scheme, the information of an input feature is encoded in the *frequency* of spikes over a time window. A higher feature value (e.g., a brighter pixel) corresponds to a higher firing rate.3 A common way to implement this is to treat the normalized input feature value as the probability of a spike occurring at each discrete time step, effectively a series of Bernoulli trials.3 While intuitive and robust to noise, rate coding can be inefficient, requiring numerous time steps and spikes to reliably represent a value.3  
* **Latency Coding:** Here, information is encoded in the *timing* of a single spike. A stronger input stimulus results in an earlier firing time.8 This method is highly efficient, as it can convey information with a maximum of one spike per neuron, making it well-suited for low-power and rapid-response applications.3  
* **Delta Modulation:** This advanced scheme generates spikes based on the *temporal change* of an input signal. A spike is fired only when the input value changes by more than a certain threshold, with separate "on-spikes" for increases and "off-spikes" for decreases.3 This method is inherently event-driven and promotes high levels of sparsity, as it filters out static information.

### **1.4 Network Architectures: Feedforward, Recurrent, and Hybrid Topologies**

Like ANNs, SNNs can be constructed in various architectures, defining the flow of information through the network graph.

* **Feedforward Networks:** This is the most straightforward architecture, where information flows strictly in one direction, from input layers to output layers, with no cycles or feedback loops.8  
* **Recurrent Networks:** In these networks, connections between neurons can form a directed graph that contains cycles, allowing for feedback. This enables the network to maintain an internal state and exhibit complex temporal dynamics.8 It is important to recognize that even the simplest spiking neuron models, like the Leaky Integrate-and-Fire neuron, possess an  
  *implicit* form of recurrence. The neuron's membrane potential at the current time step depends on its potential from the previous time step, creating a recurrent connection with itself through time. This makes SNNs fundamentally recurrent systems, poised to leverage techniques developed for training RNNs.11  
* **Hybrid Networks:** These architectures combine both feedforward and recurrent connections to create more complex and powerful computational structures. An example is the "synfire chain," a multilayered network where synchronous waves of spikes can propagate through layers.8

## **Section 2: Core Implementation: Neuron and Synapse Models**

### **2.1 Choosing a Neuron Model: A Trade-off Analysis**

The choice of the neuron model is a foundational architectural decision when building an SNN library. This decision involves a critical trade-off between computational efficiency and biological plausibility. A model that is too simple may lack the dynamic richness to solve complex problems, while one that is too complex may be computationally intractable for large-scale networks. Furthermore, the choice of model is deeply intertwined with the simulation paradigm; simpler, linear models enable highly efficient event-driven simulation, whereas complex, non-linear models necessitate a less efficient, time-stepped numerical integration approach.

The three most prominent models occupy different points on this spectrum:

* **Leaky Integrate-and-Fire (LIF):** The LIF model is the workhorse of large-scale SNN simulation due to its exceptional computational efficiency.12 It models the neuron as a simple leaky capacitor, where the membrane potential integrates incoming synaptic currents and "leaks" charge over time.9 When the potential reaches a fixed threshold, a spike is emitted, and the potential is reset.15 Its primary limitation is its simplicity; it integrates inputs linearly and cannot reproduce the complex firing patterns of biological neurons.17  
* **Izhikevich Model:** This model strikes a remarkable balance, offering much of the dynamic richness of complex biophysical models at a computational cost comparable to the LIF model.12 It uses a two-dimensional system of ordinary differential equations (ODEs) for the membrane potential and a "recovery" variable.18 By tuning just four parameters, the Izhikevich model can reproduce a wide variety of realistic neuronal behaviors, including regular spiking, bursting, and chattering.19 This makes it an excellent choice for simulations that require more biological fidelity without sacrificing large-scale performance.  
* **Hodgkin-Huxley (HH) Model:** This is the gold standard for biological accuracy. It is a detailed biophysical model that describes the dynamics of specific ion channels (e.g., sodium and potassium) that govern the action potential.23 This fidelity comes at an immense computational cost, requiring the numerical solution of several coupled, non-linear differential equations.24 The HH model is thousands of times more expensive to simulate than LIF or Izhikevich models, making it prohibitive for networks of more than a few hundred neurons and generally unsuitable for a high-performance training engine.12 It serves primarily as a benchmark for accuracy in computational neuroscience.

The following table provides a clear comparison to guide the selection of a core neuron model for the library.

| Feature | Leaky Integrate-and-Fire (LIF) | Izhikevich | Hodgkin-Huxley (HH) |
| :---- | :---- | :---- | :---- |
| **Computational Cost (FLOPs/step)** | \~5 12 | \~14 12 | \~1200 13 |
| **Biological Plausibility** | Low 15 | Medium-High 18 | Very High 23 |
| **State Variables** | 1 (Membrane Potential) | 2 (Potential, Recovery) | 4+ (Potential, Gating Variables) |
| **Key Advantage** | Extreme Speed & Simplicity | Rich Firing Dynamics | Biophysical Accuracy |
| **Primary Use Case** | Large-Scale ML/Inference | Large-Scale Brain Simulation | Single-Neuron Biophysics |

Given the goal of a high-performance training and inference engine, the **Leaky Integrate-and-Fire (LIF) model** is the most suitable choice for the core implementation due to its unparalleled computational efficiency and amenability to hardware acceleration. The **Izhikevich model** should be considered a secondary, optional implementation for users requiring more complex neuron dynamics.

### **2.2 Implementing the Leaky Integrate-and-Fire (LIF) Model: A Step-by-Step Guide**

The dynamics of an LIF neuron are described by a linear differential equation.15 For a digital simulation, this is converted into a discrete-time update rule. The change in membrane potential

V(t) is governed by the leakage of existing potential and the integration of new input current I(t).

The continuous-time differential equation is:

τm​dtdV(t)​=−(V(t)−Vrest​)+Rm​I(t)

where τm​ is the membrane time constant, Vrest​ is the resting potential, and Rm​ is the membrane resistance.9  
For a discrete-time simulation with time step Δt, this can be approximated using the Euler method, leading to the following recursive update equation:

V\[t+1\]=V\[t\]+τm​Δt​(−(V\[t\]−Vrest​)+Rm​I\[t\])

This can be simplified by defining a decay factor β=exp(−Δt/τm​), which represents the fraction of the membrane potential that remains after one time step. The updated potential is then the decayed previous potential plus the new input.  
Spiking and Reset Mechanism:  
At each time step, after the potential V\[t+1\] is updated, it is checked against the firing threshold Vth​:

1. **If V\[t+1\]≥Vth​**: The neuron fires a spike, S\[t+1\]=1. The membrane potential is then reset to a reset potential, V\[t+1\]=Vreset​.  
2. **If V\[t+1\]\<Vth​**: The neuron does not fire, S\[t+1\]=0, and the membrane potential remains at its updated value.

Below are conceptual implementations in C++ and Rust, emphasizing a data-oriented design for performance. The state of all neurons in a layer should be stored in contiguous arrays to maximize cache locality.

**C++ Implementation Sketch:**

C++

\#**include** \<vector\>

struct LIFNeuronState {  
    float voltage;  
    // Add other state variables if needed (e.g., refractory counter)  
};

class LIFLayer {  
public:  
    LIFLayer(size\_t num\_neurons, float v\_th, float v\_reset, float beta)  
        : states(num\_neurons, {0.0f}), threshold(v\_th), reset(v\_reset), decay\_beta(beta) {}

    // input\_currents and output\_spikes are pointers to GPU/CPU memory  
    void update(const float\* input\_currents, bool\* output\_spikes, size\_t n) {  
        for (size\_t i \= 0; i \< n; \++i) {  
            // 1\. Decay and Integrate  
            states\[i\].voltage \= states\[i\].voltage \* decay\_beta \+ input\_currents\[i\];

            // 2\. Check for spike  
            if (states\[i\].voltage \>= threshold) {  
                output\_spikes\[i\] \= true;  
                states\[i\].voltage \= reset;  
            } else {  
                output\_spikes\[i\] \= false;  
            }  
        }  
    }

private:  
    std::vector\<LIFNeuronState\> states;  
    float threshold;  
    float reset;  
    float decay\_beta;  
};

**Rust Implementation Sketch:**

Rust

pub struct LifNeuronState {  
    pub voltage: f32,  
}

pub struct LifLayer {  
    states: Vec\<LifNeuronState\>,  
    threshold: f32,  
    reset: f32,  
    decay\_beta: f32,  
}

impl LifLayer {  
    pub fn new(num\_neurons: usize, v\_th: f32, v\_reset: f32, beta: f32) \-\> Self {  
        LifLayer {  
            states: vec\!,  
            threshold: v\_th,  
            reset: v\_reset,  
            decay\_beta: beta,  
        }  
    }

    // Slices ensure safe, bounded memory access  
    pub fn update(&mut self, input\_currents: &\[f32\], output\_spikes: &mut \[bool\]) {  
        for i in 0..self.states.len() {  
            // 1\. Decay and Integrate  
            self.states\[i\].voltage \= self.states\[i\].voltage \* self.decay\_beta \+ input\_currents\[i\];

            // 2\. Check for spike  
            if self.states\[i\].voltage \>= self.threshold {  
                output\_spikes\[i\] \= true;  
                self.states\[i\].voltage \= self.reset;  
            } else {  
                output\_spikes\[i\] \= false;  
            }  
        }  
    }  
}

### **2.3 Simulation Paradigms: The Critical Choice Between Clock-Driven and Event-Driven Execution**

The simulation loop is the engine that drives the SNN forward in time. The choice of its architecture is one of the most critical design decisions, with profound implications for performance and accuracy. There are two primary paradigms: clock-driven and event-driven.4

* **Clock-Driven (Time-Stepped) Simulation:** This is the more straightforward approach. The simulation advances in fixed, discrete time steps (e.g., 1 ms). In every single time step, the state of *all* neurons in the network is updated, regardless of whether they received any input.4 While simple to implement and parallelize (as each neuron update is independent), this method is fundamentally inefficient for SNNs. It fails to exploit the inherent sparsity of spiking activity, wasting computational cycles on the vast majority of neurons that are silent at any given moment.7 Its accuracy is also limited by the size of the time step.25  
* **Event-Driven Simulation:** This paradigm fully embraces the event-based nature of SNNs. Instead of stepping through time, the simulation jumps directly from one spike event to the next. When a neuron fires, the simulator calculates precisely when this spike will arrive at all of its postsynaptic targets. These future spike-delivery events are placed into a time-ordered priority queue. The simulator's main loop simply pulls the next event from this queue, advances the simulation time to that event's time, and processes its consequences (i.e., updating the membrane potentials of only the affected neurons).6 This approach is maximally efficient for sparse networks, as no computation is performed during periods of silence.7 However, it is more complex to implement, especially for networks with heterogeneous synaptic delays, and can be harder to parallelize effectively.  
* **The Hybrid Approach:** An advanced strategy combines both paradigms to leverage the strengths of each.27 In a hybrid simulator, the network can be partitioned. Neuron populations that are highly active or are described by complex, non-linear models (which must be numerically integrated) can be simulated using a clock-driven method. The rest of the network, particularly sparsely active populations of simple neurons, can be simulated using an event-driven method.27 This allows for a flexible trade-off between simulation speed and model complexity.

The structure of an SNN is more than a static graph of connections; it is a dynamic, stateful system. The simulation loop is the algorithm that propagates information across this graph, updating the state of its nodes (neurons) and edges (synapses) over time. This distinction is crucial for designing data structures that can efficiently represent both the static topology and the dynamic state of all network components.

## **Section 3: Training Methodologies for SNNs**

Training an SNN involves adjusting its synaptic weights to perform a desired computation. The methods for achieving this are diverse, ranging from biologically inspired local rules to global optimization techniques adapted from deep learning. The choice of training methodology has significant implications for the library's architecture, as it dictates the types of update operations that must be supported.

### **3.1 Unsupervised Feature Learning with Spike-Timing-Dependent Plasticity (STDP)**

Spike-Timing-Dependent Plasticity (STDP) is a form of Hebbian learning that is both biologically plausible and computationally powerful for unsupervised learning.1 It operates on a simple but profound principle: the precise timing of spikes determines the change in synaptic strength.10 If a presynaptic neuron fires a few milliseconds

*before* its postsynaptic target, the synapse is strengthened in a process called Long-Term Potentiation (LTP). Conversely, if the presynaptic spike arrives *after* the postsynaptic neuron has already fired, the synapse is weakened via Long-Term Depression (LTD).10

This temporal dependency is captured by the STDP "learning window," W(Δt), where Δt=tpost​−tpre​ is the time difference between the postsynaptic and presynaptic spikes. A common mathematical formulation for this window is a pair of exponentials:  
$$ W(\\Delta t) \= \\begin{cases} A\_+ \\exp(-\\Delta t / \\tau\_+) & \\text{if } \\Delta t \> 0 \\text{ (LTP)} \\ \-A\_- \\exp(\\Delta t / \\tau\_-) & \\text{if } \\Delta t \< 0 \\text{ (LTD)} \\end{cases} $$  
where A+​ and A−​ control the magnitude of change, and τ+​ and τ−​ are the time constants of the learning window, typically in the range of tens of milliseconds.30 For efficient online implementation, this rule is often modeled using "traces" left by pre- and post-synaptic spikes, which decay over time. A weight update is triggered by a spike event and its magnitude depends on the current value of the corresponding trace from the other neuron.30  
Because STDP relies only on local spike timing information, it is an inherently **unsupervised** learning rule. It allows a network to learn statistical regularities and salient features from its input data without requiring any explicit labels or error signals.5 For example, when presented with natural images, an SNN with STDP can self-organize to develop neurons selective for intermediate-complexity visual features, such as parts of faces or objects, mimicking the function of the brain's visual cortex.32

### **3.2 Supervised Learning via Backpropagation: The Surrogate Gradient Method**

Applying the powerful gradient-based optimization methods that drive modern deep learning to SNNs presents a fundamental challenge: the non-differentiability of the spike generation mechanism.11 A neuron's output is a discontinuous Heaviside step function of its membrane potential. The derivative of this function is a Dirac delta function—zero everywhere except at the threshold, where it is infinite.36 During backpropagation, this derivative is used in the chain rule to calculate weight gradients. A zero or infinite derivative effectively stops the flow of gradient information, preventing learning.37

The **surrogate gradient (SG) method** elegantly circumvents this problem.36 The core idea is to use the exact, non-differentiable step function during the forward pass of the network to generate spikes, but to substitute a smooth, continuous, and well-behaved "surrogate" function for its derivative during the backward pass.36 This "gradient proxy" provides a usable, non-zero gradient in the vicinity of the firing threshold, allowing error signals to propagate backward through the network.

Common surrogate functions include smoothed versions of the step function's derivative, such as the derivative of the fast sigmoid or the ArcTangent function.37 The "slope" or "smoothness" of the surrogate function is a hyperparameter that can be tuned to control the learning dynamics.37

By employing surrogate gradients, SNNs can be treated as a special form of Recurrent Neural Network (RNN) and trained with the **Backpropagation Through Time (BPTT)** algorithm.11 The network is "unrolled" over its simulation time steps, and the SG method allows error gradients to be propagated backward through both the spatial layers of the network and the temporal steps of the simulation.40 Frameworks like

snnTorch are built around this principle, integrating spiking neurons as recurrent layers within a PyTorch-compatible autograd system.37

The learning rules of STDP and SG-BPTT represent two fundamentally different philosophies. STDP is a *local* rule; a synaptic update depends only on information immediately available at the synapse (pre- and post-synaptic spike times).30 This is computationally efficient and highly parallelizable. In contrast, SG-BPTT is a

*global* rule; a synaptic update depends on an error signal calculated at the network's output and propagated backward through the entire graph.11 This requires more memory and communication but allows for direct optimization on a specific task loss. A versatile SNN library must be architected to support both paradigms, perhaps with distinct

local\_update and global\_update steps in its API.

### **3.3 Bridging the Gap: High-Performance ANN-to-SNN Conversion**

A third, highly effective approach to obtaining high-performance SNNs is through **ANN-to-SNN conversion**. This is an indirect training method that leverages the maturity and power of conventional deep learning frameworks.6 The process involves first training a standard ANN (e.g., a CNN or Transformer) to high accuracy on a given task. Then, the architecture and learned weights of this ANN are systematically mapped to an equivalent SNN.42 The core principle of this mapping is the equivalence between a neuron's continuous activation value in an ANN and its average firing rate over time in an SNN.6

The primary advantage of this method is its ability to achieve state-of-the-art accuracy, often matching or exceeding what is possible with direct SNN training, especially for very deep and complex architectures like Transformers.42 It effectively bypasses the challenges of direct SNN training, such as vanishing gradients or hyperparameter tuning.

However, the conversion process is not without its own challenges. The main issue is the **conversion error**, a discrepancy between the ANN's precise analog computation and the SNN's discrete, rate-based approximation.43 This error can degrade performance and often necessitates long inference times (i.e., many simulation time steps) for the SNN's firing rates to converge and accurately represent the ANN's activations, which increases latency and energy consumption.47 A significant body of research focuses on techniques to minimize this error, such as

**threshold balancing** (adjusting neuron thresholds to match the activation statistics of the ANN) and **weight normalization**.42

This conversion technique should be viewed not just as a training method, but as a powerful network compression and deployment strategy. It provides a practical pathway for taking a highly accurate, but power-intensive, ANN model and deploying it on energy-efficient neuromorphic hardware. A comprehensive SNN library should therefore include a dedicated conversion module capable of importing pre-trained ANN models and automatically handling the complexities of weight and threshold mapping to produce a ready-to-run SNN graph.

## **Section 4: Architectural Design of the Core Library and API**

The design of a high-performance SNN engine requires careful consideration of the programming language, the public-facing API, and the internal data structures. These choices will dictate the library's performance, safety, usability, and extensibility. A robust architecture should be layered, separating the high-performance core from a stable, interoperable interface and user-friendly language bindings.

### **4.1 Language Selection Deep Dive: Rust vs. C++**

For the core engine, where performance and reliability are paramount, the choice is between low-level systems languages. C++ has long been the default for high-performance computing, but Rust presents a compelling modern alternative.

* **Performance:** Both C++ and Rust are compiled languages that offer fine-grained control over memory, leading to comparable top-tier performance.49 C++ may hold a slight advantage in certain niche benchmarks due to its decades of compiler optimization and a wider array of mature, highly-tuned numerical libraries.49  
* **Memory Safety:** This is the most significant differentiator. C++ uses manual memory management, which, while powerful, is a notorious source of bugs like buffer overflows, use-after-free errors, and memory leaks.51 Modern C++ features like smart pointers mitigate but do not eliminate these risks. Rust, by contrast, enforces memory safety at compile time through its novel ownership and borrow-checking system, preventing entire classes of memory-related bugs without the runtime overhead of garbage collection.51  
* **Concurrency:** SNN simulation is an inherently parallel problem. Rust's ownership model extends to concurrency, guaranteeing the absence of data races at compile time—a feature it calls "fearless concurrency".50 Writing safe and correct multi-threaded code in C++ remains a significant challenge for even experienced developers.  
* **Ecosystem:** C++ has a vast and mature ecosystem, particularly in scientific computing.49 Rust's ecosystem is younger but is expanding rapidly and benefits from superior tooling, including the widely praised  
  Cargo package and build manager.49

**Recommendation:** For a new, high-performance SNN engine where reliability and safe concurrency are critical, **Rust is the superior choice**. The compile-time safety guarantees will prevent subtle and catastrophic bugs that are common in large-scale, parallel C++ codebases, ultimately leading to a more robust and maintainable system.

### **4.2 Designing a Stable C-Compatible Foreign Function Interface (FFI)**

To ensure the core SNN engine can be used by a wide range of applications and programming languages (e.g., Python, C\#), it must expose its functionality through a stable Application Binary Interface (ABI). The C ABI is the de facto industry standard for cross-language interoperability.55 C++ is unsuitable for this purpose due to its unstable ABI and compiler-specific name mangling schemes.55 Therefore, the Rust core must be wrapped in a C-compatible FFI layer.

Best practices for designing this FFI layer include:

* **C Calling Convention:** All exported functions must be marked to use the C calling convention and prevented from name mangling. In Rust, this is achieved with \#\[no\_mangle\] pub extern "C" fn....56  
* **Opaque Pointers:** The internal Rust objects (e.g., Network, LifLayer) should not be exposed directly. Instead, they should be passed across the FFI boundary as opaque pointers (e.g., \*mut Network). The C client holds a handle to the object without knowing its internal layout.58  
* **Explicit Memory Management:** Since C does not have Rust's ownership system, memory must be managed manually via the API. For every object, the API must provide create\_\*, destroy\_\*, and use\_\* functions.58  
* **C-Compatible Types:** All data types in function signatures must have a defined C representation. This means using primitive types from crates like libc or cty (e.g., c\_int, c\_float) and representing arrays as a pointer and a length.55  
* **Simplicity:** The C API should be minimal and avoid complex features like C preprocessor macros to ensure maximum compatibility with different language bindings.59

This three-layer architecture—a high-performance Rust core, a stable C-FFI layer, and high-level language bindings (like a Python wrapper)—provides an ideal combination of performance, safety, and usability.

### **4.3 Core Data Structures: Representing Network Graphs and Spike Trains**

The efficiency of the SNN simulation is critically dependent on the choice of data structures for representing the network's state.

* **Graph Representation:** An SNN's synaptic connectivity is a sparse graph.  
  * An **adjacency matrix**, which uses O(V2) memory (where V is the number of neurons), is prohibitively expensive for large networks.60  
  * An **adjacency list** is more memory-efficient for sparse graphs, requiring O(V+E) memory (where E is the number of synapses).60 However, a naive implementation (e.g., a vector of linked lists) suffers from poor cache performance due to scattered memory allocations.  
  * For high performance, a data structure that ensures memory contiguity is essential. The **Compressed Sparse Row (CSR)** format is an excellent choice. It represents the graph using three contiguous arrays: one for synaptic weights, one for the column indices of the destination neurons, and a third to index the start of each neuron's outgoing connections. This layout is extremely cache-friendly for traversing a neuron's synapses.62  
* **Spike Train Representation:** The optimal way to store spikes depends on the simulation paradigm.  
  * For a **clock-driven** simulation, a simple boolean or integer array, indexed by neuron ID, is sufficient to indicate which neurons fired in the current time step.  
  * For an **event-driven** simulation, a more sophisticated structure is needed to manage future events. A **priority queue**, ordered by time, is the canonical data structure for this purpose. Each entry in the queue would represent a spike event, containing the target neuron ID, the arrival time, and the synaptic weight.

## **Section 5: GPU Acceleration and Parallelization Strategies**

To achieve the highest levels of performance for both training and inference, particularly for large networks, leveraging the massive parallelism of Graphics Processing Units (GPUs) is essential. However, efficiently mapping the asynchronous, sparse nature of SNNs onto the synchronous, data-parallel architecture of a GPU presents unique challenges. The central conflict lies in translating the asynchronous, event-driven SNN paradigm into a form that can be efficiently executed by a synchronous, Single-Instruction, Multiple-Thread (SIMT) machine like a GPU.

### **5.1 The GPU Computing Model for SNNs (CUDA/OpenCL)**

General-Purpose GPU (GPGPU) programming allows a host program running on the CPU to offload massively parallel computations to a GPU device.63 The two dominant frameworks are NVIDIA's

**CUDA** and the open standard **OpenCL**. CUDA is proprietary to NVIDIA hardware but has a more mature ecosystem and tooling for scientific computing.63 OpenCL offers portability across hardware from different vendors (NVIDIA, AMD, Intel).65 For a library targeting high-end machine learning applications, a primary focus on CUDA is pragmatic, with OpenCL as a potential extension for broader compatibility.65

The CUDA programming model organizes computation in a hierarchy of **threads**, which are grouped into **blocks**, which in turn form a **grid**.63 All threads in a block can cooperate using fast on-chip shared memory. Peak performance is achieved by adhering to key principles: minimizing data transfers between the CPU host and GPU device, maximizing the number of active threads (occupancy), and structuring memory access so that threads within a group (a "warp") access contiguous memory locations—a pattern known as

**coalesced memory access**.63

### **5.2 Memory Optimization: Leveraging Sparse Matrix Formats**

The representation of the synaptic weight matrix is the most critical data structure for GPU performance. While SNNs are sparse, GPUs excel at processing dense, regular data. The choice of sparse matrix format must balance memory efficiency with the need for regular, parallelizable access patterns.

* **Compressed Sparse Row (CSR):** This is the standard format for sparse matrices on CPUs and is very memory-efficient.68 However, on a GPU, the varying number of non-zero elements per row can lead to workload imbalance among threads in a warp. If one thread is processing a neuron with many synapses while others are processing neurons with few, the latter will sit idle, wasting computational resources.69  
* **Sliced Ellpack (SELL):** This is a GPU-optimized format designed to mitigate the load-balancing problem of CSR.68 It works by partitioning the rows of the matrix into slices of a fixed size. Within each slice, all rows are padded with dummy values to match the length of the longest row in that slice. This creates a more regular, rectangular data structure that maps much more efficiently to the SIMT execution model of GPUs, improving load balancing and enabling coalesced memory access at the cost of slightly higher memory usage due to padding.68

The choice of format depends on the target hardware. A flexible library should be capable of storing its graph in a generic format internally and then compiling it into the optimal format for the selected backend—CSR for CPU execution and SELL for GPU execution.

| Feature | Coordinate (COO) | Compressed Sparse Row (CSR) | Sliced Ellpack (SELL) |
| :---- | :---- | :---- | :---- |
| **Memory Footprint** | High | Low 68 | Medium (due to padding) 68 |
| **Spike Propagation Performance** | Low | Medium | High |
| **Load Balancing on GPU** | Poor | Poor 69 | Good 68 |
| **Suitability for Plasticity (Updates)** | Difficult | Difficult | Difficult |
| **Best For** | Initial construction | CPU Backend | GPU Backend (Inference) |

### **5.3 Parallelization Patterns: Neuron-centric vs. Synapse-centric**

There are two primary strategies for parallelizing the core computation of spike propagation and synaptic updates on a GPU 70:

* **Neuron-Parallel (or "N-algorithm"):** In this approach, one GPU thread is assigned to each *postsynaptic* neuron. Each thread is responsible for iterating through its incoming connections, checking if any presynaptic neurons have fired, and updating its own membrane potential. This approach is conceptually simple but can lead to highly inefficient, scattered memory reads as each thread accesses different parts of the synaptic weight matrix.70  
* **Synapse-Parallel (or "S-algorithm"):** This approach assigns one GPU thread to each *synapse* connected to a neuron that has just fired. When a neuron spikes, a block of threads is launched, with each thread calculating the effect of one of its outgoing synapses. This leads to much more regular, coalesced memory reads from the weight matrix. However, it introduces a new problem: multiple threads may attempt to update the membrane potential of the same postsynaptic neuron simultaneously. This requires the use of **atomic operations** to ensure thread-safe updates, which can serialize execution and become a bottleneck if many neurons converge on the same target.70

More advanced hybrid strategies exist, such as assigning a full warp of 32 threads to each firing neuron, where threads within the warp cooperate to process its outgoing synapses. The optimal strategy often depends on the specific network topology and activity levels.

### **5.4 Writing Efficient Kernels: Spike Propagation and Synaptic Updates**

A high-performance GPU implementation typically involves batching events. Instead of processing one spike at a time, all spikes occurring within a simulation time step are collected and processed together in a single, large parallel kernel launch. This transforms the sparse, asynchronous problem into a dense, synchronous one that is well-suited for the GPU.

The simulation loop on the GPU would typically consist of two main kernels:

1. **Spike Propagation Kernel:** This kernel takes as input a list of neuron IDs that fired in the previous time step. It uses a synapse-parallel approach. For each firing neuron, threads are launched to read its outgoing connections from the synaptic matrix (stored in SELL format) and calculate the input current for the target neurons. These currents are added to a temporary array using atomic operations.  
2. **Neuron State Update Kernel:** This kernel uses a neuron-parallel approach. One thread is launched for each neuron in the network. Each thread reads the total input current calculated by the first kernel, updates its neuron's membrane potential according to the LIF equations, checks for a threshold crossing, and if a spike occurs, writes its own ID to an output list for the next iteration.

Implementing plasticity (e.g., STDP) on the GPU is particularly challenging. The highly optimized, static sparse matrix formats like SELL are not designed for frequent modification. This creates a conflict between the optimal data structure for inference (static, regular) and the requirements for training (dynamic, modifiable). A practical solution is to use a more flexible data structure during training and then "compile" the trained network into a static, high-performance format for deployment and inference.

## **Section 6: Targeting Neuromorphic Hardware (NPUs)**

While GPUs offer significant acceleration for SNNs, they are ultimately general-purpose parallel processors simulating a fundamentally different computational model. Neuromorphic Processing Units (NPUs) are a class of specialized hardware designed from the ground up to execute SNNs natively, promising orders-of-magnitude improvements in energy efficiency.71 A forward-looking SNN library must be architected with these emerging platforms in mind.

### **6.1 An Overview of the Neuromorphic Landscape**

The neuromorphic field is driven by several key research and commercial chips:

* **Intel Loihi 2:** This is a state-of-the-art digital neuromorphic research chip. A single chip contains 128 neuromorphic cores and 6 x86 processor cores, supporting up to 1 million neurons and 120 million synapses.72 It is a fully asynchronous, event-driven architecture that features programmable neuron models, graded spikes (carrying integer payloads), and, crucially, support for programmable  
  **on-chip learning rules** like three-factor STDP.73  
* **IBM TrueNorth & NorthPole:** TrueNorth was a pioneering large-scale neuromorphic chip featuring 4096 cores, 1 million neurons, and 256 million synapses.74 It operates with extremely low power (tens of milliwatts) using a Globally Asynchronous, Locally Synchronous (GALS) design, where cores are internally synchronous but communicate asynchronously across a network-on-chip.75 Its successor, NorthPole, further optimizes for low-precision inference.74

### **6.2 Programming Models for NPUs**

Programming these chips requires specialized software frameworks that abstract away the complex hardware details.

* **The Lava Software Framework (for Loihi 2):** Lava is an open-source, platform-agnostic framework for developing neuro-inspired applications.73 Its architecture is built on two key abstractions:  
  * **Processes:** These are the fundamental, stateful building blocks of an application, representing components like neurons, synapses, or I/O interfaces. They communicate via event-based message passing.76  
  * **ProcessModels:** This is the concrete implementation of a Process's behavior for a specific hardware backend (e.g., a Python model for CPU simulation, a C model, or a microcode implementation for the Loihi neurocores).77

    This powerful abstraction allows developers to prototype and debug an application on a conventional CPU and then, with minimal changes, compile and deploy the same application to run on Loihi hardware.76  
* **The "Corelet" Concept (for TrueNorth):** Programming TrueNorth is based on a library of pre-defined **"corelets"**. A corelet is a self-contained program that specifies the connectivity and neuron parameters for one of the chip's 256-neuron cores, effectively representing a reusable neural computation primitive. Complex applications are built by composing and connecting these corelets.79

### **6.3 Architectural Considerations for NPU-Native SNNs**

Designing an SNN to run natively on an NPU requires a shift in mindset from pure software simulation.

* **Embracing Hardware Constraints:** Neuromorphic chips have hard physical limits, such as the maximum number of synapses per neuron or the amount of on-chip memory per core. The SNN graph topology must be designed to conform to these constraints.  
* **On-Chip vs. Off-Chip Training:** The ability of chips like Loihi 2 to perform learning directly on the hardware is a paradigm shift.72 It enables real-time adaptation and online learning, moving beyond the conventional train-on-GPU, deploy-to-hardware workflow.  
* **Communication as the Bottleneck:** On a GPU, the bottleneck is often raw computational throughput. On an NPU, where neuron updates are handled by dedicated and highly efficient circuits, the primary bottlenecks shift to the bandwidth of the on-chip communication network and the capacity of the on-chip memory.73 Therefore, optimization for NPUs focuses on designing compact network topologies, using low-precision weights, and developing algorithms that minimize overall spike traffic.

A truly practical and future-proof SNN library must be architected with a **backend abstraction layer**. The high-level user API for defining a network should be hardware-agnostic. The library's "compiler" would then be responsible for taking this abstract graph definition and translating it into the appropriate, highly-optimized representation for the selected backend: CUDA kernels and SELL-format matrices for a GPU, or a graph of Lava Processes and ProcessModels for Loihi 2\.

## **Section 7: Synthesis and Practical Recommendations**

This guide has traversed the theoretical foundations, implementation details, and hardware acceleration strategies for building a high-performance Spiking Neural Network engine. This final section synthesizes these concepts into a concrete reference architecture and provides a practical walkthrough for a sample application.

### **7.1 A Reference Architecture for a Hybrid CPU-GPU SNN Engine**

A robust, flexible, and high-performance SNN library should be designed with a layered, multi-backend architecture. This design separates concerns, allowing each component to be optimized for its specific purpose: performance, interoperability, or usability.

The proposed reference architecture consists of three primary layers built on top of multiple execution backends:

1. **Backend Layer:** This layer contains the concrete implementations for different hardware targets.  
   * **CPU Backend:** An implementation optimized for single- or multi-core CPUs, likely using a CSR graph representation and an event-driven or hybrid simulation loop.  
   * **GPU Backend:** A highly parallel implementation using CUDA, a SELL graph representation, and batched, clock-driven simulation kernels.  
   * **NPU Backend:** An interface to a neuromorphic programming framework like Lava, responsible for translating the abstract network graph into hardware-compatible Processes and ProcessModels.  
2. **Core Engine Layer (Rust):** This is the heart of the library, written in Rust for maximum performance and safety. It contains the abstract definitions of neurons, synapses, and network graphs, as well as the logic for training algorithms like STDP and Surrogate Gradient Descent. A **Backend Dispatcher** within this layer selects the appropriate backend at runtime and manages the translation of the network into the backend's specific data structures.  
3. **C-FFI Layer:** A thin, stable API that exposes the functionality of the Core Engine using the C ABI. This layer deals with raw pointers and C-compatible data types, ensuring the engine can be called from any language.  
4. **High-Level API Layer (Python):** An ergonomic, user-friendly wrapper around the FFI layer. This is the interface most users will interact with. It provides idiomatic Python classes for building networks, loading data, and running simulations, hiding the complexity of the underlying FFI calls.

This layered design allows for independent development and optimization of each component while providing a powerful and flexible system for SNN research and deployment.

### **7.2 Step-by-Step Guide to Building a Sample Application**

To illustrate the practical use of the proposed library, consider training a simple convolutional SNN to classify the MNIST dataset.

1. **Define the Network (Python API):** The user would define the network architecture using the high-level Python API. This would look very similar to defining a model in PyTorch or Keras, promoting ease of use.  
   Python  
   \# Conceptual Python API Code  
   from snn\_engine import layers, models, training

   model \= models.Sequential()

2. **Load and Encode Data:** Standard data loaders (e.g., from torchvision) would be used to load the MNIST dataset. The library's dedicated spike encoding module would then convert the static images into spike trains over a specified number of time steps (num\_steps) using rate encoding.3  
   Python  
   \# Conceptual Code  
   from snn\_engine import spikegen  
   from torchvision import datasets, transforms

   \# Standard data loading  
   train\_loader \=...

   \# Encoding data to spikes  
   for data, targets in train\_loader:  
       \# Convert batch of images to spike trains of shape  
       spiked\_data \= spikegen.rate(data, num\_steps=100)

3. **Train on GPU:** The user would configure a trainer object, specifying the model, loss function (e.g., a spike-count-based cross-entropy), optimizer, and the desired backend. Calling the fit method would trigger the Core Engine to dispatch the training task to the GPU backend.  
   Python  
   \# Conceptual Code  
   trainer \= training.Trainer(  
       model=model,  
       loss='ce\_count\_loss',  
       optimizer='adam',  
       backend='gpu' \# Select the GPU backend  
   )  
   trainer.fit(train\_loader, epochs=10)

   Internally, the library would convert the graph to the SELL format, transfer data to the GPU, and execute the BPTT algorithm using surrogate gradients and the CUDA kernels.  
4. **Inference and Decoding:** After training, the model can be used for inference. The output of the network is a series of spike trains from the final layer. A decoding function, such as selecting the neuron with the highest total spike count, is used to determine the final class prediction.11

### **7.3 Future Directions: Challenges and Opportunities**

The field of SNNs and neuromorphic computing is rapidly evolving. Several key areas present both challenges and exciting opportunities for the future:

* **Scalability:** While this guide focuses on a single-GPU architecture, the next frontier is scaling simulations to multi-GPU and multi-node clusters. This introduces significant challenges in network partitioning and efficient inter-node communication.80  
* **Advanced Learning Rules:** The development of more sophisticated and biologically plausible learning rules, such as three-factor rules that incorporate global reward signals, will be crucial. Enabling these rules to run efficiently on-chip, as supported by hardware like Loihi 2, will open new avenues for real-time learning systems.73  
* **Software-Hardware Co-design:** The most significant performance and efficiency gains will come from a tight integration of algorithm and hardware design. Future SNN models will need to be designed with the specific constraints and capabilities of target NPUs in mind, moving away from a purely software-centric approach.81  
* **Standardization and Interoperability:** As the number of SNN frameworks and hardware platforms grows, standardization becomes critical. Intermediate representations (IRs) like the Neuromorphic Intermediate Representation (NIR) are vital for allowing models trained in one framework (e.g., snnTorch) to be deployed on different hardware or simulated in another framework, fostering a more collaborative and robust ecosystem.41

#### **Works cited**

1. Spiking Neural Networks in Deep Learning \- GeeksforGeeks, accessed on September 8, 2025, [https://www.geeksforgeeks.org/deep-learning/spiking-neural-networks-in-deep-learning-/](https://www.geeksforgeeks.org/deep-learning/spiking-neural-networks-in-deep-learning-/)  
2. The Complete Guide to Spiking Neural Networks | by Amit Yadav | Biased-Algorithms, accessed on September 8, 2025, [https://medium.com/biased-algorithms/the-complete-guide-to-spiking-neural-networks-f9c1e650d69e](https://medium.com/biased-algorithms/the-complete-guide-to-spiking-neural-networks-f9c1e650d69e)  
3. Tutorial 1 \- Spike Encoding — snntorch 0.9.4 documentation, accessed on September 8, 2025, [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_1.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html)  
4. Energy-Efficient Digital Design: A Comparative Study of Event-Driven and Clock-Driven Spiking Neurons This work was partially supported by project SERICS (PE00000014) under the MUR National Recovery and Resilience Plan funded by the European Union. To foster research in this field, we are making our experimental code available as open source: https \- arXiv, accessed on September 8, 2025, [https://arxiv.org/html/2506.13268v1](https://arxiv.org/html/2506.13268v1)  
5. Unsupervised learning of digit recognition using spike-timing-dependent plasticity \- PMC, accessed on September 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4522567/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4522567/)  
6. Optimizing event-driven spiking neural network with regularization and cutoff \- Frontiers, accessed on September 8, 2025, [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1522788/full)  
7. EDHA: Event-Driven High Accurate Simulator for Spike Neural Networks \- MDPI, accessed on September 8, 2025, [https://www.mdpi.com/2079-9292/10/18/2281](https://www.mdpi.com/2079-9292/10/18/2281)  
8. Spiking Neural Network Architectures | by NeuroCortex.AI \- Medium, accessed on September 8, 2025, [https://medium.com/@theagipodcast/spiking-neural-network-architectures-e6983ff481c2](https://medium.com/@theagipodcast/spiking-neural-network-architectures-e6983ff481c2)  
9. Lab 2: The Integrate-and-Fire Model, accessed on September 8, 2025, [https://goldmanlab.faculty.ucdavis.edu/wp-content/uploads/sites/263/2016/07/IntegrateFire.pdf](https://goldmanlab.faculty.ucdavis.edu/wp-content/uploads/sites/263/2016/07/IntegrateFire.pdf)  
10. Spike-timing-dependent plasticity \- Wikipedia, accessed on September 8, 2025, [https://en.wikipedia.org/wiki/Spike-timing-dependent\_plasticity](https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity)  
11. Tutorial 5 \- Training Spiking Neural Networks with snntorch \- Read the Docs, accessed on September 8, 2025, [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_5.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html)  
12. On the Capabilities and Computational Costs of Neuron Models ..., accessed on September 8, 2025, [https://www.researchgate.net/publication/264126040\_On\_the\_Capabilities\_and\_Computational\_Costs\_of\_Neuron\_Models](https://www.researchgate.net/publication/264126040_On_the_Capabilities_and_Computational_Costs_of_Neuron_Models)  
13. Implementation of a 12-Million Hodgkin-Huxley Neuron Network on a Single Chip \- arXiv, accessed on September 8, 2025, [https://arxiv.org/pdf/2004.13334](https://arxiv.org/pdf/2004.13334)  
14. Tutorial 2 \- The Leaky Integrate-and-Fire Neuron — snntorch 0.9.4 ..., accessed on September 8, 2025, [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_2.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html)  
15. 1.3 Integrate-And-Fire Models | Neuronal Dynamics online book, accessed on September 8, 2025, [https://neuronaldynamics.epfl.ch/online/Ch1.S3.html](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)  
16. Tutorial 1: The Leaky Integrate-and-Fire (LIF) Neuron Model — Neuromatch Academy, accessed on September 8, 2025, [https://compneuro.neuromatch.io/tutorials/W2D3\_BiologicalNeuronModels/student/W2D3\_Tutorial1.html](https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial1.html)  
17. 1.4 Limitations of the Leaky Integrate-and-Fire Model | Neuronal Dynamics online book, accessed on September 8, 2025, [https://neuronaldynamics.epfl.ch/online/Ch1.S4.html](https://neuronaldynamics.epfl.ch/online/Ch1.S4.html)  
18. Izhikevich model \- Fabrizio Musacchio, accessed on September 8, 2025, [https://www.fabriziomusacchio.com/blog/2024-04-29-izhikevich\_model/](https://www.fabriziomusacchio.com/blog/2024-04-29-izhikevich_model/)  
19. Simple model of spiking neurons \- Neural Networks ... \- Izhikevich.org, accessed on September 8, 2025, [https://www.izhikevich.org/publications/spikes.pdf](https://www.izhikevich.org/publications/spikes.pdf)  
20. On the Capabilities and Computational Costs of Neuron Models \- Semantic Scholar, accessed on September 8, 2025, [https://www.semanticscholar.org/paper/On-the-Capabilities-and-Computational-Costs-of-Skocik-Long/f1e1bbe816a54f75831b9a91addd9938ccc4b2f3](https://www.semanticscholar.org/paper/On-the-Capabilities-and-Computational-Costs-of-Skocik-Long/f1e1bbe816a54f75831b9a91addd9938ccc4b2f3)  
21. Simple model of spiking neurons \- Neural Networks, IEEE Transactions on \- Washington, accessed on September 8, 2025, [https://courses.cs.washington.edu/courses/cse528/07sp/izhi1.pdf](https://courses.cs.washington.edu/courses/cse528/07sp/izhi1.pdf)  
22. Memristive Izhikevich Spiking Neuron Model and Its Application in Oscillatory Associative Memory \- Frontiers, accessed on September 8, 2025, [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.885322/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.885322/full)  
23. The Hodgkin-Huxley Heritage: From Channels to Circuits \- PMC \- PubMed Central, accessed on September 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3500626/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3500626/)  
24. Predicting Spike Features of Hodgkin-Huxley-Type Neurons With Simple Artificial Neural Network \- Frontiers, accessed on September 8, 2025, [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.800875/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.800875/full)  
25. EvtSNN: Event-driven SNN simulator optimized by population and pre-filtering \- Frontiers, accessed on September 8, 2025, [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.944262/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.944262/full)  
26. EvtSNN: Event-driven SNN simulator optimized by population and pre-filtering \- PMC \- PubMed Central, accessed on September 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9560128/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9560128/)  
27. Event and Time Driven Hybrid Simulation of Spiking Neural Networks \- ResearchGate, accessed on September 8, 2025, [https://www.researchgate.net/publication/221582286\_Event\_and\_Time\_Driven\_Hybrid\_Simulation\_of\_Spiking\_Neural\_Networks](https://www.researchgate.net/publication/221582286_Event_and_Time_Driven_Hybrid_Simulation_of_Spiking_Neural_Networks)  
28. Spike Timing Dependent Plasticity: A Consequence of More Fundamental Learning Rules \- Frontiers, accessed on September 8, 2025, [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2010.00019/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2010.00019/full)  
29. Spike timing-dependent plasticity: a Hebbian learning rule \- PubMed, accessed on September 8, 2025, [https://pubmed.ncbi.nlm.nih.gov/18275283/](https://pubmed.ncbi.nlm.nih.gov/18275283/)  
30. Spike-timing dependent plasticity \- Scholarpedia, accessed on September 8, 2025, [http://www.scholarpedia.org/article/Spike-timing\_dependent\_plasticity](http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity)  
31. Bonus Tutorial: Spike-timing dependent plasticity (STDP) — Neuromatch Academy, accessed on September 8, 2025, [https://compneuro.neuromatch.io/tutorials/W2D3\_BiologicalNeuronModels/student/W2D3\_Tutorial4.html](https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial4.html)  
32. Unsupervised Learning of Visual Features through Spike Timing Dependent Plasticity \- PMC, accessed on September 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC1797822/](https://pmc.ncbi.nlm.nih.gov/articles/PMC1797822/)  
33. Deep Unsupervised Learning Using Spike-Timing-Dependent Plasticity \- arXiv, accessed on September 8, 2025, [https://arxiv.org/html/2307.04054v2](https://arxiv.org/html/2307.04054v2)  
34. Unsupervised Learning of Visual Features through Spike Timing Dependent Plasticity | PLOS Computational Biology \- Research journals, accessed on September 8, 2025, [https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030031](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030031)  
35. Unsupervised learning using STDP \- Frontiers, accessed on September 8, 2025, [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/epub](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/epub)  
36. Surrogate Gradient Learning in Spiking Neural Networks \- arXiv, accessed on September 8, 2025, [https://arxiv.org/pdf/1901.09948](https://arxiv.org/pdf/1901.09948)  
37. Tutorial 6 \- Surrogate Gradient Descent in a Convolutional SNN \- snnTorch \- Read the Docs, accessed on September 8, 2025, [https://snntorch.readthedocs.io/en/latest/tutorials/tutorial\_6.html](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html)  
38. Differentiable Spike: Rethinking Gradient-Descent for Training Spiking Neural Networks, accessed on September 8, 2025, [https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf)  
39. Fine-Tuning Surrogate Gradient Learning for Optimal Hardware Performance in Spiking Neural Networks \- arXiv, accessed on September 8, 2025, [https://arxiv.org/html/2402.06211v1](https://arxiv.org/html/2402.06211v1)  
40. Learnable Surrogate Gradient for Direct Training Spiking Neural Networks \- IJCAI, accessed on September 8, 2025, [https://www.ijcai.org/proceedings/2023/0335.pdf](https://www.ijcai.org/proceedings/2023/0335.pdf)  
41. snnTorch Documentation — snntorch 0.9.4 documentation, accessed on September 8, 2025, [https://snntorch.readthedocs.io/](https://snntorch.readthedocs.io/)  
42. High-accuracy deep ANN-to-SNN conversion using quantization-aware training framework and calcium-gated bipolar leaky integrate and fire neuron \- PMC, accessed on September 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10030499/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030499/)  
43. A New ANN-SNN Conversion Method with High Accuracy, Low Latency and Good Robustness | IJCAI, accessed on September 8, 2025, [https://www.ijcai.org/proceedings/2023/342](https://www.ijcai.org/proceedings/2023/342)  
44. Training-Free ANN-to-SNN Conversion for High-Performance ... \- arXiv, accessed on September 8, 2025, [https://arxiv.org/abs/2508.07710](https://arxiv.org/abs/2508.07710)  
45. Towards High-performance Spiking Transformers from ANN to SNN Conversion \- arXiv, accessed on September 8, 2025, [https://arxiv.org/abs/2502.21193](https://arxiv.org/abs/2502.21193)  
46. SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN \- arXiv, accessed on September 8, 2025, [https://arxiv.org/html/2406.03470v1](https://arxiv.org/html/2406.03470v1)  
47. (PDF) Differential Coding for Training-Free ANN-to-SNN Conversion \- ResearchGate, accessed on September 8, 2025, [https://www.researchgate.net/publication/389548460\_Differential\_Coding\_for\_Training-Free\_ANN-to-SNN\_Conversion](https://www.researchgate.net/publication/389548460_Differential_Coding_for_Training-Free_ANN-to-SNN_Conversion)  
48. Inference-Scale Complexity in ANN-SNN Conversion for High-Performance and Low-Power Applications, accessed on September 8, 2025, [https://openaccess.thecvf.com/content/CVPR2025/papers/Bu\_Inference-Scale\_Complexity\_in\_ANN-SNN\_Conversion\_for\_High-Performance\_and\_Low-Power\_Applications\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Bu_Inference-Scale_Complexity_in_ANN-SNN_Conversion_for_High-Performance_and_Low-Power_Applications_CVPR_2025_paper.pdf)  
49. Rust Vs C++: Which One To Choose? \- Hashe Computer Solutions, accessed on September 8, 2025, [https://www.hashe.com/coding/rust-vs-cpp-which-to-choose/](https://www.hashe.com/coding/rust-vs-cpp-which-to-choose/)  
50. Rust Vs C++ Performance: When Speed Matters \- BairesDev, accessed on September 8, 2025, [https://www.bairesdev.com/blog/when-speed-matters-comparing-rust-and-c/](https://www.bairesdev.com/blog/when-speed-matters-comparing-rust-and-c/)  
51. Rust vs C++: Performance, Speed, Safety & Syntax Compared, accessed on September 8, 2025, [https://www.codeporting.com/blog/rust\_vs\_cpp\_performance\_safety\_and\_use\_cases\_compared](https://www.codeporting.com/blog/rust_vs_cpp_performance_safety_and_use_cases_compared)  
52. Rust vs. C++: Which Language Reigns Supreme? \- Orient Software, accessed on September 8, 2025, [https://www.orientsoftware.com/blog/rust-vs-cplusplus/](https://www.orientsoftware.com/blog/rust-vs-cplusplus/)  
53. Any main reasons/points to choose rust over c++ \- help \- The Rust Programming Language Forum, accessed on September 8, 2025, [https://users.rust-lang.org/t/any-main-reasons-points-to-choose-rust-over-c/114323](https://users.rust-lang.org/t/any-main-reasons-points-to-choose-rust-over-c/114323)  
54. Why isn't Rust used more for scientific computing? (And am I being dumb with this shape idea?) \- Reddit, accessed on September 8, 2025, [https://www.reddit.com/r/rust/comments/1jjf96y/why\_isnt\_rust\_used\_more\_for\_scientific\_computing/](https://www.reddit.com/r/rust/comments/1jjf96y/why_isnt_rust_used_more_for_scientific_computing/)  
55. A little C with your Rust \- The Embedded Rust Book, accessed on September 8, 2025, [https://docs.rust-embedded.org/book/interoperability/c-with-rust.html](https://docs.rust-embedded.org/book/interoperability/c-with-rust.html)  
56. FFI \- The Rustonomicon \- Rust Documentation, accessed on September 8, 2025, [https://doc.rust-lang.org/nomicon/ffi.html](https://doc.rust-lang.org/nomicon/ffi.html)  
57. Can rust library be used from another languages in a way c libraries do? \- Stack Overflow, accessed on September 8, 2025, [https://stackoverflow.com/questions/23781124/can-rust-library-be-used-from-another-languages-in-a-way-c-libraries-do](https://stackoverflow.com/questions/23781124/can-rust-library-be-used-from-another-languages-in-a-way-c-libraries-do)  
58. Developing C wrapper API for Object-Oriented C++ code \- Stack Overflow, accessed on September 8, 2025, [https://stackoverflow.com/questions/2045774/developing-c-wrapper-api-for-object-oriented-c-code](https://stackoverflow.com/questions/2045774/developing-c-wrapper-api-for-object-oriented-c-code)  
59. Best practices to design a C API that can be ergonomically used in Rust, accessed on September 8, 2025, [https://users.rust-lang.org/t/best-practices-to-design-a-c-api-that-can-be-ergonomically-used-in-rust/133151](https://users.rust-lang.org/t/best-practices-to-design-a-c-api-that-can-be-ergonomically-used-in-rust/133151)  
60. Implementation of Graph in C++ \- GeeksforGeeks, accessed on September 8, 2025, [https://www.geeksforgeeks.org/cpp/implementation-of-graph-in-cpp/](https://www.geeksforgeeks.org/cpp/implementation-of-graph-in-cpp/)  
61. Learn Graph Data Structure (C++). Graphs are one of the ... \- Medium, accessed on September 8, 2025, [https://medium.com/@RobuRishabh/learn-graph-data-structure-c-c505e6159f6a](https://medium.com/@RobuRishabh/learn-graph-data-structure-c-c505e6159f6a)  
62. Data structure for very large graphs : r/algorithms \- Reddit, accessed on September 8, 2025, [https://www.reddit.com/r/algorithms/comments/mvcg98/data\_structure\_for\_very\_large\_graphs/](https://www.reddit.com/r/algorithms/comments/mvcg98/data_structure_for_very_large_graphs/)  
63. Brian2CUDA: Flexible and Efficient Simulation of Spiking ... \- Frontiers, accessed on September 8, 2025, [https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2022.883700/full)  
64. Brian2CUDA: Flexible and Efficient Simulation of Spiking Neural Network Models on GPUs \- PMC \- PubMed Central, accessed on September 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9660315/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9660315/)  
65. Biologically Sound Neural Networks for Embedded Systems Using ..., accessed on September 8, 2025, [https://mobile.aau.at/publications/fehervari-2013-Biologically\_Sound\_Neural\_Networks\_for\_Embedded\_Systems\_Using\_OpenCL.pdf](https://mobile.aau.at/publications/fehervari-2013-Biologically_Sound_Neural_Networks_for_Embedded_Systems_Using_OpenCL.pdf)  
66. \[2405.02019\] Fast Algorithms for Spiking Neural Network Simulation with FPGAs \- arXiv, accessed on September 8, 2025, [https://arxiv.org/abs/2405.02019](https://arxiv.org/abs/2405.02019)  
67. Sparse Matrices and Parallel Processing on GPUs, accessed on September 8, 2025, [https://icl.utk.edu/\~hanzt/talks/SparseMatricesAndParallelProcessingOnGPUs.pdf](https://icl.utk.edu/~hanzt/talks/SparseMatricesAndParallelProcessingOnGPUs.pdf)  
68. Sparse Matrix Formats — NVPL SPARSE, accessed on September 8, 2025, [https://docs.nvidia.com/nvpl/latest/sparse/storage\_format/sparse\_matrix.html](https://docs.nvidia.com/nvpl/latest/sparse/storage_format/sparse_matrix.html)  
69. Sparse GPU Kernels for Deep Learning \- People @EECS, accessed on September 8, 2025, [https://people.eecs.berkeley.edu/\~matei/papers/2020/sc\_sparse\_gpu.pdf](https://people.eecs.berkeley.edu/~matei/papers/2020/sc_sparse_gpu.pdf)  
70. Dynamic parallelism for synaptic updating in GPU-accelerated spiking neural network simulations \- PMC \- PubMed Central, accessed on September 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6147227/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6147227/)  
71. What Is Neuromorphic Computing? \- IBM, accessed on September 8, 2025, [https://www.ibm.com/think/topics/neuromorphic-computing](https://www.ibm.com/think/topics/neuromorphic-computing)  
72. Intel Loihi2 Neuromorphic Processor : Architecture & Its Working \- ElProCus, accessed on September 8, 2025, [https://www.elprocus.com/intel-loihi2-neuromorphic-processor/](https://www.elprocus.com/intel-loihi2-neuromorphic-processor/)  
73. A Look at Loihi 2 \- Intel \- Neuromorphic Chip \- Open Neuromorphic, accessed on September 8, 2025, [https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/](https://open-neuromorphic.org/neuromorphic-computing/hardware/loihi-2-intel/)  
74. Cognitive computer \- Wikipedia, accessed on September 8, 2025, [https://en.wikipedia.org/wiki/Cognitive\_computer](https://en.wikipedia.org/wiki/Cognitive_computer)  
75. TrueNorth: A Deep Dive into IBM's Neuromorphic Chip Design, accessed on September 8, 2025, [https://open-neuromorphic.org/blog/truenorth-deep-dive-ibm-neuromorphic-chip-design/](https://open-neuromorphic.org/blog/truenorth-deep-dive-ibm-neuromorphic-chip-design/)  
76. Lava Software Framework — Lava documentation, accessed on September 8, 2025, [https://lava-nc.org/](https://lava-nc.org/)  
77. Lava Tutorial: Mnist Training On Gpu And Evaluation On Loihi2 | R Gaurav's Blog, accessed on September 8, 2025, [https://r-gaurav.github.io/2024/04/13/Lava-Tutorial-MNIST-Training-on-GPU-and-Evaluation-on-Loihi2.html](https://r-gaurav.github.io/2024/04/13/Lava-Tutorial-MNIST-Training-on-GPU-and-Evaluation-on-Loihi2.html)  
78. ProcessModels — Lava documentation, accessed on September 8, 2025, [https://lava-nc.org/lava/notebooks/in\_depth/tutorial03\_process\_models.html](https://lava-nc.org/lava/notebooks/in_depth/tutorial03_process_models.html)  
79. Neuromorphic – Silverton Consulting, accessed on September 8, 2025, [https://silvertonconsulting.com/tag/neuromorphic/](https://silvertonconsulting.com/tag/neuromorphic/)  
80. Multi-GPU SNN Simulation with Static Load Balancing \- ICS-FORTH, accessed on September 8, 2025, [https://users.ics.forth.gr/\~argyros/mypapers/2021\_02\_arxiv\_bautembach.pdf](https://users.ics.forth.gr/~argyros/mypapers/2021_02_arxiv_bautembach.pdf)  
81. Neuromorphic Principles for Efficient Large Language Models on Intel Loihi 2 \- arXiv, accessed on September 8, 2025, [https://arxiv.org/html/2503.18002v2](https://arxiv.org/html/2503.18002v2)  
82. Spiking Neural Network (SNN) Frameworks \- Open Neuromorphic, accessed on September 8, 2025, [https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/](https://open-neuromorphic.org/neuromorphic-computing/software/snn-frameworks/)