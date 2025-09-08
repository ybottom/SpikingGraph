Title: Build core DLL/API (Rust cdylib)

Description:
Finalize the Rust cdylib core implementation and ensure the C-compatible header and build artifacts are produced for consumption by other languages. Provide versioned API and clear ABI guarantees.

Acceptance criteria:
- `rust_core` builds reliably in CI for Windows and Linux.
- Header `rust_core/spikinggraph.h` is kept in-sync with exported symbols.
- Example program demonstrates loading the Rust DLL and running a small network.
- Release artifact published by CI for download.

Labels: feature, core, rust
Estimated effort: 3d
Assignee:
Status: done
