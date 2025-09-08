#pragma once

#ifdef _WIN32
  #ifdef SPKG_EXPORTS
    #define SPKG_API __declspec(dllexport)
  #else
    #define SPKG_API __declspec(dllimport)
  #endif
#else
  #define SPKG_API
#endif

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* SNNHandle;

typedef void (*SpikeCallback)(int neuron_id, double time, void* user_data);

// Configuration placeholder
struct SNNConfig {
    int num_neurons;
    double dt;
};

SPKG_API SNNHandle snn_create_network(const SNNConfig* config);
SPKG_API int snn_load_weights(SNNHandle snn, const char* filename);
SPKG_API int snn_run_step(SNNHandle snn, double dt);
SPKG_API int snn_inject_current(SNNHandle snn, int neuron_id, float current);
SPKG_API int snn_add_connection(SNNHandle snn, int src_neuron, int dst_neuron, float weight);
SPKG_API int snn_register_spike_callback(SNNHandle snn, SpikeCallback cb, void* user_data);
SPKG_API int snn_get_membrane_potentials(SNNHandle snn, float* buffer, int length);
SPKG_API int snn_destroy_network(SNNHandle snn);

#ifdef __cplusplus
}
#endif
