# [SGLang] kt capture buffer sizes ignore per-request token width — segfault with speculative decoding + CUDA graph + kt

**Version**: SGLang 0.5.18 + kt-kernel 0.7.0.post2
**Symptom**: With `--speculative-algorithm STANDALONE` (or any spec mode), decode CUDA graphs enabled, and kt expert offload active, the server boots and answers single requests, then segfaults on the first target-verify graph replay under load.

**Root cause**: `decode_cuda_graph_runner` registers kt's pinned capture buffers via `KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)`. kt keys those buffers by FLAT token count. In spec mode each request contributes `captured_req_width` (= num_draft_tokens) tokens, so the flat sizes seen at capture are `bs * width`, none of which are in the registered list. kt then serves capture-time calls from its single reusable temp buffer; the next batch-size capture reallocates that temp buffer, and the recorded graph replays a host-side copy into freed pinned memory → segfault.

**Fix (3 lines)**: register width-scaled sizes:
```python
KTMoEWrapper.set_capture_batch_sizes(
    sorted({b * self.captured_req_width for b in self.capture_bs})
)
```
Verified: spec+graph+kt survives sustained load after the fix (Qwen3-Coder-480B, TP4, H100).
