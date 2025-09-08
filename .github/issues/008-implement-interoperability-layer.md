Title: Implement interoperability layer (buffer sharing, callbacks)

Description:
Implement the interoperability layer that enables zero-copy buffer sharing between host and core, reliable callback registration across languages, and explicit memory management APIs.

Acceptance criteria:
- APIs for allocating shared buffers and mapping them in host memory.
- Safe callback registration/unregistration across FFI boundaries.
- Examples showing Python/C# sharing buffers with the engine without copying.

Labels: feature, interoperability
Estimated effort: 4d
Assignee:
Status: done
