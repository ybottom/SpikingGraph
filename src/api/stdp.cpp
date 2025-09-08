// STDP learning rule implementation for SpikingGraph
// This file provides a C++ function to apply STDP updates to synaptic weights based on spike timing.

#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {

// STDP parameters (can be made configurable)
struct STDPParams {
    float A_plus = 0.01f;
    float A_minus = 0.012f;
    float tau_plus = 20.0f;
    float tau_minus = 20.0f;
};

// Per-neuron spike traces for STDP
struct STDPTraces {
    std::vector<float> pre_trace;
    std::vector<float> post_trace;
};

// Update traces and apply STDP to a single synapse
void spkg_stdp_update(
    int pre, int post,
    float& weight,
    STDPTraces& traces,
    const STDPParams& params,
    bool pre_spike, bool post_spike, float dt)
{
    // Decay traces
    traces.pre_trace[pre] *= std::exp(-dt / params.tau_plus);
    traces.post_trace[post] *= std::exp(-dt / params.tau_minus);
    // Update traces on spikes
    if (pre_spike) traces.pre_trace[pre] += 1.0f;
    if (post_spike) traces.post_trace[post] += 1.0f;
    // Apply weight update
    if (pre_spike) weight += params.A_plus * traces.post_trace[post];
    if (post_spike) weight -= params.A_minus * traces.pre_trace[pre];
    // Clamp weight
    weight = std::clamp(weight, 0.0f, 10.0f);
}

} // extern "C"
