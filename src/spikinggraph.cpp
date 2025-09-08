#include <queue>
#include <tuple>
struct Event {
    double time;
    int neuron_id;
    float weight;
    bool operator<(const Event& other) const {
        // Reverse for min-heap
        return time > other.time;
    }
};
#include "spikinggraph.h"
#include <vector>
#include <cstring>
#include <mutex>

enum NeuronModelType {
    LIF = 0,
    Izhikevich = 1,
};

struct SNNContext {
    int num_neurons;
    double dt;
    int neuron_model; // 0 = LIF, 1 = Izhikevich
    // LIF parameters per neuron
    std::vector<float> membrane_potentials;
    std::vector<float> membrane_threshold;
    std::vector<float> membrane_reset;
    std::vector<float> membrane_tau;
    // Izhikevich parameters/state
    std::vector<float> izh_v, izh_u, izh_a, izh_b, izh_c, izh_d;
    // synaptic weights (sparse adjacency list)
    std::vector<std::vector<std::pair<int,float>>> connections;
    // event-driven input queue (per-neuron accumulated current)
    std::vector<float> input_current;
    // Event-driven
    std::priority_queue<Event> event_queue;
    double sim_time = 0.0;
    double synaptic_delay = 1.0; // ms, fixed for now

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
    ctx->neuron_model = (config->neuron_model == 1) ? Izhikevich : LIF;
    // LIF
    ctx->membrane_potentials.assign(ctx->num_neurons, 0.0f);
    ctx->membrane_threshold.assign(ctx->num_neurons, 1.0f);
    ctx->membrane_reset.assign(ctx->num_neurons, 0.0f);
    ctx->membrane_tau.assign(ctx->num_neurons, 10.0f);
    // Izhikevich
    ctx->izh_v.assign(ctx->num_neurons, -65.0f);
    ctx->izh_u.assign(ctx->num_neurons, 0.0f);
    ctx->izh_a.assign(ctx->num_neurons, 0.02f);
    ctx->izh_b.assign(ctx->num_neurons, 0.2f);
    ctx->izh_c.assign(ctx->num_neurons, -65.0f);
    ctx->izh_d.assign(ctx->num_neurons, 8.0f);
    // Common
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
    double end_time = ctx->sim_time + dt;
    // Process all events up to end_time
    while (!ctx->event_queue.empty() && ctx->event_queue.top().time <= end_time) {
        Event ev = ctx->event_queue.top(); ctx->event_queue.pop();
        ctx->sim_time = ev.time;
        int i = ev.neuron_id;
        ctx->input_current[i] += ev.weight;
        if (ctx->neuron_model == LIF) {
            float V = ctx->membrane_potentials[i];
            float tau = ctx->membrane_tau[i];
            float I = ctx->input_current[i];
            float dV = (-V / tau + I); // treat event as impulse
            float Vn = V + dV + I;
            ctx->input_current[i] = 0.0f;
            if (Vn >= ctx->membrane_threshold[i]) {
                Vn = ctx->membrane_reset[i];
                if (ctx->callback) ctx->callback(i, ctx->sim_time, ctx->user_data);
                for (auto &pr : ctx->connections[i]) {
                    int dst = pr.first;
                    float w = pr.second;
                    if (dst >= 0 && dst < ctx->num_neurons) {
                        ctx->event_queue.push(Event{ctx->sim_time + ctx->synaptic_delay, dst, w});
                    }
                }
            }
            ctx->membrane_potentials[i] = Vn;
        } else if (ctx->neuron_model == Izhikevich) {
            float v = ctx->izh_v[i];
            float u = ctx->izh_u[i];
            float a = ctx->izh_a[i];
            float b = ctx->izh_b[i];
            float c = ctx->izh_c[i];
            float d = ctx->izh_d[i];
            float I = ctx->input_current[i];
            float dv = (0.04f * v * v + 5.0f * v + 140.0f - u + I);
            float du = (a * (b * v - u));
            float vn = v + dv;
            float un = u + du;
            ctx->input_current[i] = 0.0f;
            if (vn >= 30.0f) {
                vn = c;
                un += d;
                if (ctx->callback) ctx->callback(i, ctx->sim_time, ctx->user_data);
                for (auto &pr : ctx->connections[i]) {
                    int dst = pr.first;
                    float w = pr.second;
                    if (dst >= 0 && dst < ctx->num_neurons) {
                        ctx->event_queue.push(Event{ctx->sim_time + ctx->synaptic_delay, dst, w});
                    }
                }
            }
            ctx->izh_v[i] = vn;
            ctx->izh_u[i] = un;
        }
    }
    ctx->sim_time = end_time;
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
