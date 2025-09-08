This folder contains a Rust-based core library built as a cdylib exposing a C-compatible API.

Build:

- Ensure you have Rust toolchain installed (stable).
- Run: `cargo build --release` to produce a DLL (Windows) or .so (Linux) in `target/release`.

The library exposes functions like `snn_create_network`, `snn_run_step`, `snn_inject_current`, and others for consumption from C/C++ or other languages via FFI.
