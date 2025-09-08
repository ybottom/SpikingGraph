# Spike Encoding Module for SpikingGraph

This module provides functions to encode real-valued data (e.g., images, audio) into spike trains using various strategies:

- Rate coding
- Latency coding
- Delta modulation

The module is designed to be used from both C++ and Rust, and will be exposed via the C API for use in Python and other languages.

## API Sketch

- `spkg_encode_rate(const float* data, int length, int num_steps, float* out_spikes)`
- `spkg_encode_latency(const float* data, int length, int num_steps, float* out_spikes)`
- `spkg_encode_delta(const float* data, int length, int num_steps, float* out_spikes)`

Each function fills `out_spikes` with a binary (0/1) spike train for each input value over `num_steps` time steps.

## Implementation Plan

- Add C++ and Rust implementations for each encoding method
- Expose as C API functions
- Add tests and examples
