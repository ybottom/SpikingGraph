Title: Create FFI bindings for Python and Rust

Description:
Provide Python bindings (ctypes or cffi) and first-class Rust crate wrappers that expose the stable C API in idiomatic ways. Include examples and tests.

Acceptance criteria:
- `python/` package that wraps the C API using `ctypes` or `cffi` with examples.
- Rust crate providing safe wrappers around the raw FFI.
- Tests that run the Python example and Rust example using the built core.

Labels: feature, bindings, python, rust
Estimated effort: 4d
Assignee:
Status: open
