#include "spike_encoding.h"
#include <cmath>
#include <algorithm>
#include <random>

extern "C" {

// Rate coding: each value is the probability of a spike at each time step
int spkg_encode_rate(const float* data, int length, int num_steps, float* out_spikes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < length; ++i) {
        std::bernoulli_distribution d(std::clamp(data[i], 0.0f, 1.0f));
        for (int t = 0; t < num_steps; ++t) {
            out_spikes[i * num_steps + t] = d(gen) ? 1.0f : 0.0f;
        }
    }
    return 0;
}

// Latency coding: value determines the time step of the spike (lower value = earlier spike)
int spkg_encode_latency(const float* data, int length, int num_steps, float* out_spikes) {
    for (int i = 0; i < length; ++i) {
        int spike_time = static_cast<int>((1.0f - std::clamp(data[i], 0.0f, 1.0f)) * (num_steps - 1));
        for (int t = 0; t < num_steps; ++t) {
            out_spikes[i * num_steps + t] = (t == spike_time) ? 1.0f : 0.0f;
        }
    }
    return 0;
}

// Delta modulation: spike when value changes by threshold
int spkg_encode_delta(const float* data, int length, int num_steps, float* out_spikes, float threshold) {
    for (int i = 0; i < length; ++i) {
        float prev = data[i];
        out_spikes[i * num_steps + 0] = 0.0f;
        for (int t = 1; t < num_steps; ++t) {
            float curr = data[i]; // For static input, this is always the same; for time series, pass a buffer
            float diff = curr - prev;
            out_spikes[i * num_steps + t] = (std::abs(diff) > threshold) ? 1.0f : 0.0f;
            prev = curr;
        }
    }
    return 0;
}

} // extern "C"
