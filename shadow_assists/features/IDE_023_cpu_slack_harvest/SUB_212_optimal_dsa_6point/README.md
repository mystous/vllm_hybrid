# SUB_212 — Optimal+DSA 6-Point Coverage + Host DSA WQ Confounder Finding

> **status**: ✅ 완료 (2026-06-11)
> **parent**: `IDE_023` (CPU Slack Harvesting) / `TSK_043` (Host-Side Slack Reclamation)
> **predecessor**: `TSK_042` (vanilla + suffix baseline 70 cells × 2 = 140)

## 측정 범위

- **10 모델** (Qwen 7B/32B/72B, DeepSeek distill 7B/32B/70B, Llama 8B/70B/405B-FP8, DeepSeek-R1 671B)
- **7 corpus** (sharegpt/swebench/humaneval/mbpp/wildchat/lmsys/mix)
- **6 measurement points** per (model, corpus):
  - ① van(OFF), ② van(ON), ③ DSA(ON), ④ suf(OFF), ⑤ suf(ON), ⑥ suf+dsa(ON)
- **결과: 406/420 = 96.7% 측정**, 9/10 모델 6/6 완전 커버리지

## Key Findings

### 1. Host DSA WQ enable 시점이 baseline 차이의 진짜 원인

| 시점 | sysfs mtime | WQ state | Llama-8B mix vanilla tps |
|---|---|---|---:|
| 2026-06-02 (TSK_042) | (enable 이전) | DISABLED | **8,850** |
| 2026-06-08 00:40 | `wq0.0/state` mtime | enabled | — |
| 2026-06-10+ (본 sweep) | (enable 후) | ENABLED | **12,089** (+36%) |

**vllm 코드/LHC 무관 확정** — `verify_dsa.sh` 의 C1/C2/C3 (현 vllm + LHC 제거 / 옛 vllm + LHC / 옛 vllm + LHC 제거) 모두 baseline ±0.7% 동일.

### 2. 호스트 DSA 의 method 별 차등 효과 (mix corpus)

| Method | OFF→ON 효과 | 메커니즘 |
|---|---:|---|
| Vanilla | **+33~+36%** | host-bound regime, DSA memcpy 가속 도움 |
| Suffix | **−5~+10%** | step-bound regime, 효과 작음 |

### 3. vllm-level DSA env 의 영향 = 0 (noise)

- vanilla → DSA(vllm env on): 0% (Llama-8B 동일 세션)
- suffix → suffix_dsa: ±5% corpus-dependent
- **호스트 DSA 가 driver, vllm env 는 부차적**

### 4. Llama-405B-FP8 의 suffix 부팅 한계

- **Engine init failure**: `num_gpu_blocks=0 → override=512` 후 core proc crash
- **Affected**: ⑤ ⑥ — 14 cells 영구 미측정
- **Cause**: 405B-FP8 + suffix K=32 + B200 단일 TP=8 + gmu 0.85 호환성 한계

### 5. R1-671B TP=8 단일 부팅 성공

TSK_042 의 2×TP4 + gmu 0.95 setup 과 달리, **TP=8 + gmu 0.85 단일 인스턴스** 로도 4 config 모두 부팅 가능.

## 산출물 (단일 완결 문서)

| 문서 | 용도 |
|---|---|
| [`FULL_MATRIX_6point.md`](FULL_MATRIX_6point.md) | **single-doc reviewer 자료** (HW/SW/corpus/모델/Δ분해/winner/limitations 모두 포함) |
| [`OPTIMAL_DSA_70cells_flat.md`](OPTIMAL_DSA_70cells_flat.md) | 70-row flat 표 (model × corpus × 6 points + winner) |
| [`MEASUREMENTS_6point.md`](MEASUREMENTS_6point.md) | aggregate 결과 (커버리지 + per-model 표 + effect 분해) |
| [`vanilla_vs_suffix_full_matrix.md`](vanilla_vs_suffix_full_matrix.md) | vanilla×suffix only (전 sweep 산출) |
| [`MEASUREMENTS.md`](MEASUREMENTS.md) | multi-model 첫 sweep (Llama-8B 28 cells + 7 models dsa+suffix_dsa) |

## 측정 산출물

| 파일 | 내용 |
|---|---|
| `runs/summ_<TAG>_<METHOD>_<CORPUS>.json` (269 cells) | per-cell measurement |
| `runs/per_request_raw.jsonl` | per-request raw (재집계용) |
| `runs/_logs/<TAG>_<METHOD>_{boot,bench}.log` | 진행 로그 |
| `runs_synthetic_aborted/` | 초기 합성 sweep 폐기 archive (TSK_042 와 corpus 불일치 발견 후 중단) |

## Sweep scripts

| 스크립트 | 측정 범위 |
|---|---|
| [`sweep_corpus.sh`](sweep_corpus.sh) | Llama-8B × 4 configs × 7 corpus (28 cells, 첫 sweep) |
| [`sweep_multi.sh`](sweep_multi.sh) | 7 models × dsa + suffix_dsa × 7 corpus (98 cells) |
| [`sweep_complete.sh`](sweep_complete.sh) | 9 models × {missing configs} × 7 corpus (154 cells, 6/6 완성) |
| [`verify_dsa.sh`](verify_dsa.sh) | C1/C2/C3 LHC + vllm 격리 검증 (호스트 DSA confounder 확정) |
| [`verify_host_dsa.sh`](verify_host_dsa.sh) | 호스트 WQ disable 검증 (root 권한 필요로 실행 못함, 사용자 수동 실행용) |

## Aggregator

[`aggregate.py`](aggregate.py) — 6-point 매트릭스 → MEASUREMENTS_6point.md 자동 생성

## Wall time

- 첫 sweep (Llama-8B 28 cells, sweep_corpus.sh): 1h37m
- Multi-model sweep (7 models dsa+suffix_dsa, sweep_multi.sh): 5h33m
- Complete coverage sweep (9 models missing configs, sweep_complete.sh): **16h35m**
- verify_dsa (C1/C2/C3 1셀씩): ~25min
- **누적 GPU 시간**: ~24h

## Tracing

- id_registry: SUB_212 (다음 부여 번호 SUB_213)
- Predecessor IDs: TSK_042 (vanilla+suffix baseline), SUB_201 (host-path 재정립), SUB_202~211 (10 lever PoCs)
- Related: IDE_022 (AGSD Realistic Eval, TSK_043 의 parent), IDE_023 (HPC Multi-Axis, 본 SUB 의 parent)
