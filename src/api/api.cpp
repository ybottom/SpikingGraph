#include "api.h"
#include <iostream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// This file optionally loads the Rust cdylib at runtime and resolves symbols.
// If loading fails, it falls back to linking against the native C++ implementations.

static bool dynamic_loaded = false;
// function pointers for dynamic symbols
typedef void* (*fn_create)(const void*);
typedef int (*fn_destroy)(void*);
typedef int (*fn_run_step)(void*, double);
typedef int (*fn_inject)(void*, int, float);
typedef int (*fn_add_conn)(void*, int, int, float);
typedef int (*fn_register_cb)(void*, void(*)(int,double,void*), void*);
typedef int (*fn_get_pots)(void*, float*, int);

static fn_create dyn_create = nullptr;
static fn_destroy dyn_destroy = nullptr;
static fn_run_step dyn_run_step = nullptr;
static fn_inject dyn_inject = nullptr;
static fn_add_conn dyn_add_conn = nullptr;
static fn_register_cb dyn_register_cb = nullptr;
static fn_get_pots dyn_get_pots = nullptr;

static void resolve_symbols(void* handle) {
#ifdef _WIN32
    auto sym = (void*)GetProcAddress((HMODULE)handle, "snn_create_network");
    dyn_create = (fn_create)sym;
    dyn_destroy = (fn_destroy)GetProcAddress((HMODULE)handle, "snn_destroy_network");
    dyn_run_step = (fn_run_step)GetProcAddress((HMODULE)handle, "snn_run_step");
    dyn_inject = (fn_inject)GetProcAddress((HMODULE)handle, "snn_inject_current");
    dyn_add_conn = (fn_add_conn)GetProcAddress((HMODULE)handle, "snn_add_connection");
    dyn_register_cb = (fn_register_cb)GetProcAddress((HMODULE)handle, "snn_register_spike_callback");
    dyn_get_pots = (fn_get_pots)GetProcAddress((HMODULE)handle, "snn_get_membrane_potentials");
#else
    dyn_create = (fn_create)dlsym(handle, "snn_create_network");
    dyn_destroy = (fn_destroy)dlsym(handle, "snn_destroy_network");
    dyn_run_step = (fn_run_step)dlsym(handle, "snn_run_step");
    dyn_inject = (fn_inject)dlsym(handle, "snn_inject_current");
    dyn_add_conn = (fn_add_conn)dlsym(handle, "snn_add_connection");
    dyn_register_cb = (fn_register_cb)dlsym(handle, "snn_register_spike_callback");
    dyn_get_pots = (fn_get_pots)dlsym(handle, "snn_get_membrane_potentials");
#endif
}

int spkg_load_dynamic_backend(const char* path) {
    if (!path) return -1;
#ifdef _WIN32
    HMODULE h = LoadLibraryA(path);
    if (!h) {
        std::cerr << "Failed to load backend: " << path << "\n";
        return -1;
    }
    resolve_symbols((void*)h);
#else
    void* h = dlopen(path, RTLD_NOW);
    if (!h) {
        std::cerr << "Failed to load backend: " << dlerror() << "\n";
        return -1;
    }
    resolve_symbols(h);
#endif
    dynamic_loaded = true;
    return 0;
}

// C API wrappers that dispatch to dynamic backend if loaded, else to static implementations
extern "C" {

void* spkg_snn_create_network(const SNNConfig* cfg) {
    if (dynamic_loaded && dyn_create) return dyn_create((const void*)cfg);
    return snn_create_network(cfg);
}

int spkg_snn_destroy_network(void* h) {
    if (dynamic_loaded && dyn_destroy) return dyn_destroy(h);
    return snn_destroy_network(h);
}

int spkg_snn_run_step(void* h, double dt) {
    if (dynamic_loaded && dyn_run_step) return dyn_run_step(h, dt);
    return snn_run_step(h, dt);
}

int spkg_snn_inject_current(void* h, int nid, float cur) {
    if (dynamic_loaded && dyn_inject) return dyn_inject(h, nid, cur);
    return snn_inject_current(h, nid, cur);
}

int spkg_snn_add_connection(void* h, int s, int d, float w) {
    if (dynamic_loaded && dyn_add_conn) return dyn_add_conn(h, s, d, w);
    return snn_add_connection(h, s, d, w);
}

int spkg_snn_register_spike_callback(void* h, SpikeCallback cb, void* ud) {
    if (dynamic_loaded && dyn_register_cb) return dyn_register_cb(h, cb, ud);
    return snn_register_spike_callback(h, cb, ud);
}

int spkg_snn_get_membrane_potentials(void* h, float* buf, int len) {
    if (dynamic_loaded && dyn_get_pots) return dyn_get_pots(h, buf, len);
    return snn_get_membrane_potentials(h, buf, len);
}

} // extern C
