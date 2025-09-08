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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* SNNHandle;

typedef void (*SpikeCallback)(int neuron_id, double time, void* user_data);

typedef struct SNNConfig {
    int32_t num_neurons;
    double dt;
} SNNConfig;

SPKG_API SNNHandle snn_create_network(const SNNConfig* config);
SPKG_API int snn_destroy_network(SNNHandle handle);
SPKG_API int snn_run_step(SNNHandle handle, double dt);
SPKG_API int snn_inject_current(SNNHandle handle, int neuron_id, float current);
SPKG_API int snn_add_connection(SNNHandle handle, int src, int dst, float weight);
SPKG_API int snn_register_spike_callback(SNNHandle handle, SpikeCallback cb, void* user_data);
SPKG_API int snn_get_membrane_potentials(SNNHandle handle, float* buffer, int length);

#ifdef __cplusplus
}
#endif
