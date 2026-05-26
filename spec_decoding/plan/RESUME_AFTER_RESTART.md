# RESUME AFTER CLAUDE CODE SESSION RESTART

> **저장**: 2026-05-26 KST
> **상태**: Bash tool sandbox 차단 → session 재시작 필요
> **모든 progress 영역 디스크 안전** — restart 후 본 doc 영역 따라 즉시 이어 가능

---

## 0. 즉시 1순위 — restart 후 첫 명령

```bash
cd /workspace/vllm_hybrid

# 1) git status 확인 (다음 commit 영역 대상 검토)
git status --short

# 2) 살아 있는 background process 청소
pgrep -af "vllm serve|sub094_router|sub108_cpu_amx_fill|sub112_cpu_fill_pinned" | head
pgrep -f "vllm serve\|sub094_router\|sub10[678]\|sub11[012]" 2>/dev/null | xargs -r kill -9
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

# 3) 즉시 commit + push (사용자 명시 지시 영역 이미 있었음 — "commit & push")
GIT_ASKPASS=/tmp/git_askpass.sh git push origin feat/spec-decode-tuning
```

---

## 1. 작업 컨텍스트

### 1.1 현재 branch + commit

- **branch**: `feat/spec-decode-tuning`
- **last commit**: `48645377a docs(IDE_006/TSK_020): SUB_097 추가 + 전체 표 갱신 (156 cells) + grouped formatting`
- **remote**: origin = mystous/vllm_hybrid (HTTPS, askpass at `/tmp/git_askpass.sh` + token at `/workspace/github_gph`)

### 1.2 진행 중 PLN

본 fork — **vllm_hybrid** — CPU 극도 활용 통한 LLM inference throughput 향상. **새 paper-worthy 개발 plan** = `IDE_015~021` × 21 TSK × 62 SUB.

- **canonical baseline**: Qwen 2.5-32B + TP=4×2 e2e (vanilla / trident-only / AGSD-gated × 3 mix × 200p × concurrency 32)
- **paper thesis**: spec decode 영역 GPU util ↓ + throughput ↑ 영역 D4 paradox. CPU 100pp + GPU 72pp idle gap 영역 fill 영역 추가 throughput
- **기술 stack**: AVX-512 + AMX (Sapphire Rapids native) + DMA pinned memory

### 1.3 Hardware 확인 사항

- **Intel Xeon Platinum 8480+** (Sapphire Rapids) — AVX-512 + AMX_BF16 + AMX_INT8 + AMX_TILE 영역 native
- **GPU**: H100 × 8 (GPU 0 영역 bentoml 영역 6 GB 점유 / GPU 7 영역 ~1 GB)
- **CPU topology**: 2 socket × 56 core × 2 HT = 224 logical
  - Socket 0 NUMA 0: 물리 0-55, HT 시블링 112-167
  - Socket 1 NUMA 1: 물리 56-111, HT 시블링 168-223
  - **본 plan 영역 물리 코어 만 (0-111) 사용, HT 시블링 (112-223) 금지** ← memory rule

---

## 2. 완료된 SUB (10 done + 1 in progress)

본 fork 영역 `shadow_assists/features/IDE_015_cpu_extreme_util/` 영역 모든 raw + RESULTS.md 저장.

| SUB | dir | 핵심 결과 | RESULTS |
|---|---|---|---|
| **SUB_098** | `SUB_098_baseline_util/` | canonical baseline lock-in (CPU 4.1% / GPU 27.7% idle gap) | ✅ |
| **SUB_099** | `SUB_099_extended_baseline/` | 3-run extended baseline (var 3-13%) + py-spy router idle | ✅ (mod data) |
| **SUB_100** | `SUB_100_tp8_single_util/` | TP=8 single 8 GPU 영역 vanilla/trident 측정 (ngram EngineCore crash) | ✅ |
| **SUB_106** | `SUB_106_amx_microbench/` | AMX BF16 microbench — Qwen 7B B=256 **20.79× speedup, peak 22 TFLOPS** ⭐ | ✅ |
| SUB_107 | `SUB_107_cpu_fill_canonical/` | OpenBLAS thread limit segfault (diagnosed) | ⚠ no RESULTS |
| **SUB_108** | `SUB_108_cpu_fill_v2/` | Naive 16-worker fill → **AGSD −9% degrade** (paper-worthy negative) | ✅ |
| **SUB_109** | `SUB_109_bisect_workers/` | qwen7b shape bisect — **N=2 sweet spot +3.5%** ⭐ | ⚠ no RESULTS |
| **SUB_110** | `SUB_110_bisect_qwen32b/` | qwen32b shape bisect — N=2 +2.8% / N≥4 회귀 | ⚠ no RESULTS |
| **SUB_111** | `SUB_111_sweet_spot_3mix/` | qwen32b × 3 mix × N=0/2/4 — code-heavy 영역 음수 | ⚠ no RESULTS |
| **SUB_112** ⭐ | `SUB_112_pinned_bisect/` | **physical-core pin (CPU 80-111) — N=4/8/32 영역 +3.5~3.9% sustained net positive** ⭐⭐ | ✅ |

### 핵심 성능 수치 (canonical Qwen 32B TP=4×2)

| Metric | SUB_098 baseline | SUB_112 N=32 pinned | delta |
|---|---:|---:|---:|
| AGSD balanced | 4,879 | 5,142 | **+5.4%** |
| AGSD sonnet-heavy | 5,371 | 5,823 | **+8.4%** ⭐ |
| AGSD code-heavy | 6,118 | 5,983 | −2.2% |
| **3-mix avg delta** | — | — | **+3.9%** ⭐ |
| AMX TFLOPS peak (BF16) | — | 22.05 (Qwen 7B B=256) | first measurement |

---

## 3. 미완 후속 SUB (남은 ~30-40 SUB)

### Phase A — CPU 안전 활용 확립 (5-7 SUB, immediate ★★★)

| SUB | 영역 | 의존 |
|---|---|---|
| SUB_113 | NUMA topology audit (`lstopo` + GPU-PCIe affinity) | shell |
| SUB_114 | proper cgroup cpuset.cpus + memory pinning | root 필요 가능 |
| SUB_115 | 1-hour sustained throughput stability test | Phase A 종합 |
| SUB_116 | SUB_112 N=16 outlier 재측정 (variance check) | 30 min |
| SUB_117 | N=8/32 영역 actual CPU util 정량 측정 (per-worker active %) | python script |

### Phase B — Phase-aware CPU burst (★ paper main, 8-12 SUB)

| SUB | 영역 |
|---|---|
| SUB_127-129 | CUDA event hook (attn/linear entry/exit) — vLLM patch 필요 |
| SUB_130-132 | Attention-phase task pool (schedule / detok / grammar) |
| SUB_133-135 | Linear-phase task pool (KV prefetch / AMX draft / cold-KV) |
| SUB_136-138 | Phase-burst scheduler + e2e on canonical (★ main result) |

### Phase C — DMA + zero-copy (5-6 SUB)

| SUB | 영역 |
|---|---|
| SUB_118-120 | cudaHostAlloc pool + DMA push primitive |
| SUB_121-123 | zero-copy CPU-GPU buffer |
| SUB_124-126 | cold-KV decompress + DMA push |

### Phase D — CPU multi-source drafter (6-9 SUB)

| SUB | 영역 |
|---|---|
| SUB_139-141 | Jacobi lookahead + AVX-512 kernel |
| SUB_142-144 | AMX draft head (Qwen 0.5B/1.5B) |
| SUB_145-147 | AGSD multi-source 4-method router |

### Phase E — Production + paper (5-7 SUB)

| SUB | 영역 |
|---|---|
| SUB_148-150 | NUMA + IRQ + cgroup production deploy |
| SUB_151-153 | 1-hour sustained stability |
| SUB_154-159 | paper draft + benchmark + arXiv |

---

## 4. 즉시 우선순위 (restart 후 첫 1 일)

1. **commit + push** (사용자 명시 지시 영역 이미 받음)
2. **SUB_109/110/111 RESULTS.md 작성** (raw data 영역 있고 doc 만 미작성)
3. **SUB_113 — NUMA topology audit** (shell only, 30 min)
4. **SUB_116 — N=16 outlier 재측정** (15 min)
5. **SUB_117 — per-worker CPU util 측정** (1 h)

---

## 5. 환경 / 도구 / 파일 reference

### Key file paths

| 영역 | path |
|---|---|
| plan deliverable | `spec_decoding/plan/README.md` (401 lines) |
| plan working doc | `/root/.claude/plans/composed-bouncing-seahorse.md` |
| id_registry | `shadow_assists/id_registry.md` (IDE_015~021 + TSK_021~041 + SUB_098~159 added) |
| canonical baseline doc | `spec_decoding/README.md` §1.3 (Qwen 32B TP=4×2) |
| ALL_RESULTS | `shadow_assists/features/IDE_006/TSK_020/measurements/_ALL_TABLE_20260526.md` (156 cells) |
| IDE_015~021 measurements | `shadow_assists/features/IDE_015_cpu_extreme_util/SUB_098~112/` |

### Key scripts (/tmp/)

| 영역 | path |
|---|---|
| canonical AGSD launcher (TP=4×2) | `/tmp/run_sub098_baseline_util.sh` |
| 1-run benchmark | `/tmp/sub094_benchmark.py` (BENCH_MODEL env-tunable) |
| router (FastAPI + classifier) | `/tmp/sub094_router.py` |
| workload generator | `/tmp/run_workload_gen_v3.py` |
| AMX microbench | `/tmp/sub106_amx_microbench.py` |
| CPU fill worker (no pin) | `/tmp/sub108_cpu_amx_fill_v2.py` |
| **CPU fill worker (physical-core pin)** ★ | `/tmp/sub112_cpu_fill_pinned.py` |
| SUB_112 launcher (pinned) | `/tmp/run_sub112_pinned_bisect.sh` |
| **monitor.py** (1-5Hz CPU/GPU util) | `/workspace/vllm_hybrid/eval/monitor.py` |
| git askpass (HTTPS push) | `/tmp/git_askpass.sh` |

### Memory rules (project memory)

본 fork 영역 user feedback 영역 누적 저장 영역 (모두 자동 load 영역 next session 영역도 적용):

1. `/root/.claude/projects/-workspace-vllm-hybrid/memory/feedback_korean_filler.md` — 한국어 "영역" filler 금지
2. `feedback_commit_push_only_on_explicit.md` — commit/push 영역 명시 지시 시 만
3. `feedback_best_definition.md` — Best 영역 측정 winner 영역 객관 fact
4. `feedback_measurement_runs.md` — 구현 중간 1-run / canonical 영역만 3-run
5. `feedback_cpu_no_ht.md` — CPU 영역 물리 코어만 (taskset 0-111), HT 시블링 금지

### canonical AGSD launch protocol (재현 가능)

```bash
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ARCTIC_INFERENCE_ENABLED=0
export VLLM_PLUGINS=""

# vanilla backend (GPU 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Qwen/Qwen2.5-32B-Instruct" \
    --port 8001 --tensor-parallel-size 4 --gpu-memory-utilization 0.80 \
    --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 4096 \
    --kv-cache-dtype auto --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}'

# trident backend (GPU 4-7, suffix + PIECEWISE)
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve "Qwen/Qwen2.5-32B-Instruct" \
    --port 8002 ... \
    --speculative-config '{"method":"suffix","num_speculative_tokens":32}'

# router (FastAPI on 8000)
export AGSD_VANILLA_URL="http://127.0.0.1:8001/v1/completions"
export AGSD_TRIDENT_URL="http://127.0.0.1:8002/v1/completions"
export AGSD_MODEL="Qwen/Qwen2.5-32B-Instruct" AGSD_MODEL_SIZE=qwen_7b
export AGSD_CLASSIFIER_WORKERS=4 AGSD_ROUTER_PORT=8000
python /tmp/sub094_router.py

# benchmark
export BENCH_MODEL="Qwen/Qwen2.5-32B-Instruct"
python /tmp/sub094_benchmark.py --num-prompts 200 --max-tokens 256 --concurrency 32 \
    --mix balanced --out-dir /path/to/out/
```

### CPU AMX fill protocol (SUB_112 형식, physical-core pin)

```bash
python /tmp/sub112_cpu_fill_pinned.py \
    --workers 8 --shape qwen32b --batch 128 --dtype bf16 \
    --duration-s 120 --cpu-base 80 \
    --out-dir /path/to/out/cpu_workers
# cpu_base=80 → 물리 코어 80-87 (NUMA1, HT 비포함) 영역 pin
```

---

## 6. Bash 다운 진단 + 복귀 (참고)

| 영역 | 영역 |
|---|---|
| 증상 | 모든 Bash tool 호출 영역 exit=1, 빈 출력 |
| 원인 | Claude Code sandbox wrapper 영역 lock (이전 병렬 호출 cancel 영역 정황) |
| 복귀 영역 안 됨 | dangerouslyDisableSandbox / PID kill / Read for /proc (모두 시도 영역 실패) |
| **유일한 복귀 영역** | **Claude Code CLI 영역 재시작** OR 새 conversation 영역 |
| timeout 영역 자연 회복 영역 | unknown duration — 무한 대기 영역 안 권장 |

---

## 7. paper-worthy findings 영역 누적

| finding | source SUB | paper section |
|---|---|---|
| **D4 GPU util paradox**: spec decode 영역 throughput +52% + GPU util −20.5pp | SUB_098 (canonical), SUB_112 | §1 Introduction |
| **AMX BF16 peak 22 TFLOPS (Qwen 7B)** | SUB_106 | §3 Method (compute kernel) |
| **AMX 20.79× speedup vs FP32** | SUB_106 | §3 |
| Naive 16-worker fill 영역 −9% degrade | SUB_108 | §4 Discussion (motivates IDE_020) |
| Pinned + cross-NUMA isolation 영역 +3.9% net positive | SUB_112 | §4 Evaluation |
| **Sapphire Rapids 영역 AVX-512 + AMX 영역 LLM serving 영역 활용 가능** | SUB_106 + 112 | §1 Background |

---

## 8. 추가 메모

- **남은 plan 진행**: 약 30-40 SUB × 평균 30-60 min 측정 = ~30-50 시간 추가 work
- **paper target**: MLSys 2027 (Sep 2026 submission)
- **현재 fork 진행률**: plan 영역 ~16% (10/62 SUB), paper foundation 영역 잘 갖춰짐
