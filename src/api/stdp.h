#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct STDPParams {
    float A_plus;
    float A_minus;
    float tau_plus;
    float tau_minus;
};

struct STDPTraces {
    float* pre_trace;
    float* post_trace;
};

void spkg_stdp_update(
    int pre, int post,
    float* weight,
    STDPTraces* traces,
    const STDPParams* params,
    bool pre_spike, bool post_spike, float dt);

#ifdef __cplusplus
}
#endif
