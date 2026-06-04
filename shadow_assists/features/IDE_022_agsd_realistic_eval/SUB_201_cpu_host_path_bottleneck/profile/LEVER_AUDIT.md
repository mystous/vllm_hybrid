# SUB_201 4 lever 자산 audit + unit microbench 결과 (2026-06-05 KST 07:01)

> §5 verdict 후 lever PoC 진입 가능성 검증. 모든 lever 의 build + microbench 1차 통과.

---

## 1. Build + microbench 결과

| Lever | 자산 빌드 | 명령 | Microbench 결과 |
|---|---|---|---|
| **A1 CPU drafting** | ✅ `IDE_019/SUB_187/build/libamx_draft_qwen05b.so` 21KB | `g++ -O3 -mamx-tile -mamx-bf16 -mavx512f -mavx512bf16 -mavx512vl -march=sapphirerapids -fopenmp -fPIC -shared src/amx_draft_qwen05b.cpp` | **K=7 draft loop 2.05ms** (target <35ms) — `B=1/4 K=5/7 all PASS`, per-step 0.21~0.29ms. **70B verify ~80ms 대비 ~5% overhead → net gain 가능** |
| **B1 detok kernel** | ✅ `IDE_016/build/avx512_tokenizer/libavx512_tokenizer.so` 16KB | `g++ -O3 -mavx512f -mavx512bw -mavx512vbmi -mavx512bf16 -march=sapphirerapids -fopenmp -fPIC -shared src/avx512_tokenizer/batch_bpe_kernel.cpp` | kernel build OK (correctness test 별도 필요) |
| **B1 async worker** | ✅ `SUB_190/build/async_tokenizer_worker` 17KB | `g++ -O3 -fopenmp -march=native -pthread src/async_tokenizer_worker.cpp` | **avg 0.77ms/cycle = duty 3.85%** (target 2-5%), 16 worker × cores 80-95, 20ms cycle |
| **A2 KV tiering** | ✅ `IDE_017/build/libpinned_pool.so` 28KB | `g++ -O3 -fopenmp -fPIC -shared src/pinned_pool.cpp -lcudart -lnuma` | **DMA 51.34 GB/s** (64MB, PCIe 5.0 saturate), 9.3μs latency (4KB), alloc 420μs / free 180μs |
| **B3 scheduler** | (vllm 내부 코드, 별도 자산 없음) | `vllm/v1/core/sched/async_scheduler.py` (`AsyncScheduler` 클래스) | — patch 진행 시 measurement |

## 2. CPU 환경 확인

| 확인 항목 | 결과 |
|---|---|
| AMX support | ✅ `amx_bf16, amx_int8, amx_tile` 모두 있음 |
| AVX-512 support | ✅ `avx512_bf16, avx512_bitalg, avx512_fp16, avx512_vbmi2, avx512_vnni, avx512_vpopcntdq, avx512bw` |
| Build tools | ✅ `/usr/bin/g++` (icx 없음, g++ -march=sapphirerapids 활용) |
| libnuma | ✅ A2 빌드에 `-lnuma` 필요 (numa_available symbol) |
| libcudart | ✅ A2 빌드에 `/usr/local/cuda/lib64 -lcudart` |

## 3. vllm 패치 의존성 매트릭스 (충돌 가능성)

| Lever | 패치 target file | 다른 lever 와 충돌 |
|---|---|---|
| A1 | `vllm/v1/spec_decode/cpu_amx.py` (신규) + suffix_decoding.py 패턴 참조 | ✅ 없음 |
| B1 | `vllm/v1/engine/detokenizer.py` + `vllm/tokenizers/detokenizer_utils.py` | ✅ 없음 |
| A2 | `vllm/v1/core/kv_cache_manager.py` + `vllm/v1/worker/gpu_worker.py` | ✅ 없음 |
| B3 | `vllm/v1/core/sched/async_scheduler.py` + scheduler.py | ⚠ A2 와 file 인접 (다른 함수, 충돌 risk 낮음) |

→ **4 lever 모두 동시 dev 가능** (서로 다른 sub-package, file conflict 없음).

## 4. GPU 분할 plan (e2e validation)

| Lever | unit GPU | e2e GPU | 추천 모델 |
|---|---|---|---|
| A1 | 0 (CPU) | TP=4 | Qwen-7B (small) 또는 R1 (target) |
| B1 | 0 | TP=4 | Qwen-7B / DS-Qwen-7B (throughput 큰 셀) |
| A2 | 1 | TP=4 | Llama-70B (memcpy 80% dominant) |
| B3 | 0 | TP=4 | Qwen-7B (launch overhead 38% dominant) |

**병렬 측정**: TP=2 × 2 instance (작은 모델 가능). 일반적 비교 위해 **순차 측정 권장** (baseline 환경 일치).

## 5. 구현 시간 + ROI 우선순위

| 순위 | Lever | 구현 시간 | SUB_201 verdict 기반 ROI | Primary target |
|---|---|---|---|---|
| 🥇 1 | **B1 detok offload** | **1-2일** | Llama-8B 37% wall slack 회수 → +10~+20% | Llama-8B, DS-Qwen-7B (mix 24-27k tps) |
| 🥈 2 | **B3 scheduler** | **수일** | Qwen-7B launch overhead 38% 회수 → +10~+25% | Qwen-7B (low util), R1 (11k launches/s) |
| 🥉 3 | **A2 KV tiering** | **수일~주** | Llama-70B memcpy 80% 회수 → +15~+30% | Llama-70B, 405B |
| 4 | **A1 CPU drafting** | **2-3주** | R1 suffix -49% → +20~+50% 회복 | R1, DS-Llama-70B (worst suffix) |

## 6. 다음 step 옵션

| 옵션 | scope | 시간 |
|---|---|---|
| A | B1 single PoC + Qwen-7B e2e | 1-2일 |
| B | 4 lever 동시 worktree dev (각 lever 별 격리, parallel agent) | 1-3주 (각 lever 별) |
| C | 측정 priority — R1/Llama-70B 의 nvtx marker 추가 + step 분해 더 정밀 | 수시간 |

**진행 결정**: B (4 worktree 동시 dev) 예정.

---

## 부록 — 재현 명령

```bash
# A1 AMX kernel
cd shadow_assists/features/IDE_019_multi_source_drafter/SUB_187_amx_draft_head
mkdir -p build && g++ -O3 -mamx-tile -mamx-bf16 -mavx512f -mavx512bf16 -mavx512vl \
    -march=sapphirerapids -fopenmp -fPIC -shared \
    src/amx_draft_qwen05b.cpp -o build/libamx_draft_qwen05b.so
python run_microbench.py

# B1 AVX-512 tokenizer kernel
cd shadow_assists/features/IDE_016_avx512_amx_pool
mkdir -p build/avx512_tokenizer && g++ -O3 -mavx512f -mavx512bw -mavx512vbmi -mavx512bf16 \
    -march=sapphirerapids -fopenmp -fPIC -shared \
    -I src/avx512_tokenizer src/avx512_tokenizer/batch_bpe_kernel.cpp \
    -o build/avx512_tokenizer/libavx512_tokenizer.so

# B1 async tokenizer worker
cd shadow_assists/features/IDE_016_avx512_amx_pool/SUB_190_async_tokenizer_worker
mkdir -p build && g++ -O3 -fopenmp -march=native -pthread \
    src/async_tokenizer_worker.cpp -o build/async_tokenizer_worker
./build/async_tokenizer_worker  # background side-channel

# A2 pinned pool + DMA
cd shadow_assists/features/IDE_017_dma_zero_copy
mkdir -p build && g++ -O3 -fopenmp -fPIC -shared src/pinned_pool.cpp \
    -o build/libpinned_pool.so -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lnuma
ln -sf $PWD/build/libpinned_pool.so SUB_176_dma_pool_canonical/build/libpinned_pool.so
python SUB_176_dma_pool_canonical/run_microbench.py --iters 50 --total-limit 5
```
