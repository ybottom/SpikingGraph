Title: Develop NPU backend and graceful fallback

Description:
Add hooks and an adapter layer to target NPUs (e.g., Intel/Amd NPUs, Lava/Loihi integration). Provide an API that can dispatch to vendor SDKs if available and gracefully fallback to CPU/GPU.

Acceptance criteria:
- Backend abstraction interface for NPUs.
- Adapter for at least one NPU-like interface or mock demonstrating the flow.
- Graceful fallback logic documented and tested.

Labels: feature, backend, npu
Estimated effort: 8d
Assignee:
Status: open
