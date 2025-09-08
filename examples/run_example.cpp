#include "spikinggraph.h"
#include <iostream>

void my_spike_callback(int neuron_id, double time, void* user_data) {
    std::cout << "Neuron " << neuron_id << " spiked at " << time << "\n";
}

int main() {
    SNNConfig cfg;
    cfg.num_neurons = 10;
    cfg.dt = 0.001;
    // Select neuron model: 0 = LIF, 1 = Izhikevich
    cfg.neuron_model = 1; // Try 0 for LIF, 1 for Izhikevich

    SNNHandle h = snn_create_network(&cfg);
    snn_register_spike_callback(h, my_spike_callback, nullptr);

    // create a simple chain: 0 -> 1 -> 2
    snn_add_connection(h, 0, 1, 1.2f);
    snn_add_connection(h, 1, 2, 1.0f);

    // inject current into neuron 0 at t=0
    snn_inject_current(h, 0, 2.0f);

    for (int i = 0; i < 50; ++i) {
        snn_run_step(h, cfg.dt);
    }

    float buffer[10];
    int copied = snn_get_membrane_potentials(h, buffer, 10);
    std::cout << "Copied " << copied << " potentials\n";

    snn_destroy_network(h);
    return 0;
}
