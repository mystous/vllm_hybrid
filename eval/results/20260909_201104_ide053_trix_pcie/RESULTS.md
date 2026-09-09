# IDE_053 — TriX-on-PCIe 교차점 측정 — **중단 (사용자 지시, 부분 결과)**

경로 A (pinned 호스트 → H2D 복사 → GPU 언팩 → matmul, 4 GPU 가 expert 의 1/4 shard 를 동시에 처리) 만 측정됨. 경로 B (zero-copy) 는 cupy 14 의 `runtime.hostGetDevicePointer` 부재로 실패 (미측정).

| m (행) | GPU 경로 µs/expert (4 GPU 병렬) | per-GPU GB/s | CPU AMX/AVX 실측 µs/expert |
|---|---|---|---|
| 1~8 | 792~819 | 7.2~7.4 | 66~120 |
| 16 / 32 | 819 / 816 | 7.2 | 212 / 495 |
| 64 / 128 | 798 / 803 | 7.4 | ~1000 (vec) / 275 (AMX) |
| 256 / 512 / 1024 | 799 / 805 / 794 | 7.3 | 500 / — / — |

주의: per-GPU 7.3 GB/s 는 PCIe Gen5 x16 의 상한 (~50 GB/s) 에 크게 못 미침. 측정 경로가 numpy 뷰 → `torch.from_numpy().copy_()` 라 torch 가 pageable 로 취급해 스테이징 복사가 끼었을 가능성이 큼 (측정 방법의 한계). 따라서 이 표는 상한 판정에 쓸 수 없고, zero-copy·cudaMemcpyAsync(pinned) 로 재측정해야 함.
현재 수치 기준으로는 모든 m 에서 GPU 경로 (~800 µs) 가 CPU 경로보다 느리며, m=64 (vec 경로) 에서만 근접. **결론 보류.**
