#include "spikinggraph.h"
#include <vector>
#include <cstring>
#include <mutex>

struct SNNContext {
    int num_neurons;
    double dt;
    // LIF parameters per neuron
    std::vector<float> membrane_potentials;
    std::vector<float> membrane_threshold;
    std::vector<float> membrane_reset;
    std::vector<float> membrane_tau;
    // synaptic weights (sparse adjacency list)
    std::vector<std::vector<std::pair<int,float>>> connections;
    // event-driven input queue (per-neuron accumulated current)
    std::vector<float> input_current;

    SpikeCallback callback;
    void* user_data;
    std::mutex mtx;
};

extern "C" {

SNNHandle snn_create_network(const SNNConfig* config) {
    if (!config) return nullptr;
    SNNContext* ctx = new SNNContext();
    ctx->num_neurons = config->num_neurons;
    ctx->dt = config->dt;
    ctx->membrane_potentials.assign(ctx->num_neurons, 0.0f);
    ctx->membrane_threshold.assign(ctx->num_neurons, 1.0f);
    ctx->membrane_reset.assign(ctx->num_neurons, 0.0f);
    ctx->membrane_tau.assign(ctx->num_neurons, 10.0f);
    ctx->connections.resize(ctx->num_neurons);
    ctx->input_current.assign(ctx->num_neurons, 0.0f);
    ctx->callback = nullptr;
    ctx->user_data = nullptr;
    return static_cast<SNNHandle>(ctx);
}

int snn_load_weights(SNNHandle snn, const char* filename) {
    // Stub: no-op for now
    return 0;
}

int snn_run_step(SNNHandle snn, double dt) {
    if (!snn) return -1;
    SNNContext* ctx = static_cast<SNNContext*>(snn);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    // LIF dynamics (Euler integration) with event-driven inputs
    for (int i = 0; i < ctx->num_neurons; ++i) {
        float I = ctx->input_current[i];
        float V = ctx->membrane_potentials[i];
        float tau = ctx->membrane_tau[i];
        // dV/dt = -V/tau + I
        float dV = (-V / tau + I) * static_cast<float>(dt);
        V += dV;
        ctx->membrane_potentials[i] = V;
        // reset input after consumed
        ctx->input_current[i] = 0.0f;
        if (V >= ctx->membrane_threshold[i]) {
            // spike
            ctx->membrane_potentials[i] = ctx->membrane_reset[i];
            if (ctx->callback) ctx->callback(i, 0.0, ctx->user_data);
            // propagate to targets
            for (auto &pr : ctx->connections[i]) {
                int dst = pr.first;
                float w = pr.second;
                if (dst >= 0 && dst < ctx->num_neurons) {
                    ctx->input_current[dst] += w;
                }
            }
        }
    }
    return 0;
}

int snn_inject_current(SNNHandle snn, int neuron_id, float current) {
    if (!snn) return -1;
    SNNContext* ctx = static_cast<SNNContext*>(snn);
    if (neuron_id < 0 || neuron_id >= ctx->num_neurons) return -1;
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->input_current[neuron_id] += current;
    return 0;
}

int snn_add_connection(SNNHandle snn, int src_neuron, int dst_neuron, float weight) {
    if (!snn) return -1;
    SNNContext* ctx = static_cast<SNNContext*>(snn);
    if (src_neuron < 0 || src_neuron >= ctx->num_neurons) return -1;
    if (dst_neuron < 0 || dst_neuron >= ctx->num_neurons) return -1;
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->connections[src_neuron].push_back({dst_neuron, weight});
    return 0;
}

int snn_register_spike_callback(SNNHandle snn, SpikeCallback cb, void* user_data) {
    if (!snn) return -1;
    SNNContext* ctx = static_cast<SNNContext*>(snn);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->callback = cb;
    ctx->user_data = user_data;
    return 0;
}

int snn_get_membrane_potentials(SNNHandle snn, float* buffer, int length) {
    if (!snn || !buffer) return -1;
    SNNContext* ctx = static_cast<SNNContext*>(snn);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    int to_copy = std::min(length, ctx->num_neurons);
    std::memcpy(buffer, ctx->membrane_potentials.data(), to_copy * sizeof(float));
    return to_copy;
}

int snn_destroy_network(SNNHandle snn) {
    if (!snn) return -1;
    SNNContext* ctx = static_cast<SNNContext*>(snn);
    delete ctx;
    return 0;
}

} // extern "C"
