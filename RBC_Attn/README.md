# RBC-Attn v0.1 — Reduction-Budgeted Coalescing Attention

명세: `docs/RBC_Attn_design.md` (설계), `docs/AGENT_TASK.md` (검증 지시).

이미 주어진 block-sparse attention 을 실행하는 **작업 분해·결합 알고리즘**의 프로토타입이다.
CPU 가 희소 연결을 분석해 GPU 작업을 재구성하고, GPU 가 실제 QK/softmax/PV 를 수행한다.
양자화·압축·pruning·선택 KV 삭제를 하지 않으며, 원래 sparse edge 집합을 정확히 보존한다.

## 빌드와 실행

```bash
bash build.sh                       # C++/OpenMP 플래너 빌드
bash run_host_tests.sh              # 호스트 테스트 (CPU only)
python -m pytest -q tests/test_gpu.py   # GPU 정확성 (CUDA 필요)
bash run_gpu.sh                     # 벤치 일괄
```

## 구조

| 파일 | 기능 |
|---|---|
| `src/planner.cpp` | C++/OpenMP exact-edge compiler |
| `rbc/planner.py` | 플래너 ctypes 바인딩 + 파이썬 참조 구현 |
| `rbc/kernels.py` | Triton multi-block attention + partial merge |
| `rbc/runtime.py` | pinned descriptor upload, CUDA workspace |
| `rbc/native.py` | FlashInfer native paged-attention 대조 |
| `rbc/data.py` | synthetic mask / 실제 capture 저장 |
| `bench.py` | 동일 executor 비교, native 비교, fresh/run-only |
| `stream_bench.py` | 준비된 query chunk 의 CPU/GPU 동시 실행 |
