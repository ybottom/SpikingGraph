#pragma once

#include "../../include/spikinggraph.h"

// This header exposes a stable C API for higher-level language bindings.
// It forwards directly to the core implementation (Rust cdylib if present, else C++ static/shared lib).

#ifdef __cplusplus
extern "C" {
#endif

// Reuse the types from include/spikinggraph.h

// Helper to load dynamic backend (optional)
int spkg_load_dynamic_backend(const char* path);

#ifdef __cplusplus
}
#endif
