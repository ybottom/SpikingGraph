#ifdef __cplusplus
extern "C" {
#endif

// STDP learning rule API
typedef struct STDPParams {
  float A_plus;
  float A_minus;
  float tau_plus;
  float tau_minus;
} STDPParams;

typedef struct STDPTraces {
  float* pre_trace;
  float* post_trace;
} STDPTraces;

SPKG_API void spkg_stdp_update(
  int pre, int post,
  float* weight,
  STDPTraces* traces,
  const STDPParams* params,
  int pre_spike, int post_spike, float dt);

#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif

// Spike encoding API
SPKG_API int spkg_encode_rate(const float* data, int length, int num_steps, float* out_spikes);
SPKG_API int spkg_encode_latency(const float* data, int length, int num_steps, float* out_spikes);
SPKG_API int spkg_encode_delta(const float* data, int length, int num_steps, float* out_spikes, float threshold);

#ifdef __cplusplus
}
#endif
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


enum NeuronModelType {
  LIF = 0,
  Izhikevich = 1,
};

struct SNNConfig {
  int num_neurons;
  double dt;
  int neuron_model; // 0 = LIF, 1 = Izhikevich
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
