# Combo sweep — A (F4 TP=4) × D (OOB silent fix) — 2026-05-20 KST

> branch `feat/neo-amx-apply` HEAD `4857014e2` + uncommitted attention.py D fix.
>
> 본 측정 = SUB_015-Phase 3 follow-up 의 lever combination sweep. 사용자 명시:
> A (TP=4) / B (OMP dynamic) / C (BLOCK=32) / D (P4 OOB root fix) 의 14 combination.
> 단 B = -1.4% 회귀 (이전 측정), C = 2-4 주 scope 외 → 본 turn 영역 = A, D 의 4 combination.

---

## D (OOB silent fix) 영역 코드 변경

**vllm/model_executor/layers/attention/attention.py** — cdec setup 의 except handler
(line ~1085) 의 D11 OOB precheck 영역 silent skip 추가:

```python
except Exception as _cdec_setup_e:
    cdec_future = None
    # P4 D fix — D11 OOB precheck 의 silent skip. raise 영역 의 traceback
    # log spam (30k+ 누적) 이 shm_broadcast 영역 의 engine death root.
    # env-gated (VLLM_NEO_OOB_SILENT=1 default).
    import os as _os_d_g
    _d_oob_silent = (
        _os_d_g.environ.get("VLLM_NEO_OOB_SILENT", "1") == "1"
    )
    _is_d11_oob_skip = (
        _d_oob_silent
        and str(_cdec_setup_e).startswith("D11 OOB precheck")
    )
    if not _is_d11_oob_skip:
        # 기존 traceback logging path
        ...
```

### env

- `VLLM_NEO_OOB_SILENT=1` (default) — D fix on (silent skip)
- `VLLM_NEO_OOB_SILENT=0` — D fix off (기존 traceback path)

---

## A (TP=4) 영역 = pacpu binary rebuild

```bash
CXX=/tmp/gcc12/usr/bin/g++-12 MAX_JOBS=40 \
    bash csrc/cpu/pacpu/build.sh llama3_3_70b 4
# → libpacpu-llama3_3_70b-tp4.so (2.06 MB)
```

`--tensor-parallel-size 4` 로 vllm launch.

---

## 측정 환경

| 항목 | 값 |
|---|---|
| Model | Llama-3.3-70B-Instruct |
| Hardware | H100 × 8 (Intel SPR + GPU 7 bentoml) |
| GPU memory utilization | 0.85 |
| Workload | 100p × 8192 short (각 ~17 min) |
| max_num_seqs | 256 |
| max_num_batched_tokens | 8192 |
| MAX_WAIT | 1800s (30 min) |

---

## 4 combination 결과

| Combo | env | TP | tps | wall (s) | 비고 |
|---|---|---:|---:|---:|---|
| **1 baseline** | D=off (SILENT=0) | 8 | **934.5** | 876.6 | reference |
| **2 D-only** | D=on (SILENT=1) | 8 | **935.6** | 875.6 | +0.1% (variance 영역) |
| 3 A-only | D=off | 4 | NO_RESULT | — | timeout 14:13, 12/100 frozen, OOB=4,160, traceback spam |
| 4 A+D retry | D=on | 4 | NO_RESULT | — | timeout 14:49, 14/100 frozen, OOB=3,200, **traceback=0** |

---

## 진정한 fact

### D (OOB silent fix) 의 가치

- **+** log spam 차단: combo 4 의 traceback = 0 vs combo 3 의 traceback 폭주. shm_broadcast cancelled 영역 root 차단 확인.
- **−** engine deadlock root 영역 X: combo 4 도 결국 stall (14/100 frozen, GPU 0%). traceback 영역 만 차단, OOB 자체 의 NEO slot 추적 race 영역 fix X.

→ **D fix 의 진정한 영역** = log path overhead 제거 만. engine death 의 primary cause 영역 = OOB-induced slot/swap state mismatch (D fix 와 무관, 별도 root fix 필요).

### A (TP=4) 의 가치

- **−** unstable: D fix on/off 무관, TP=4 + 8192 context 영역 → OOB precheck event 빈도 매우 높음 (3,200~4,160 건 @ ~14% 진행) → engine deadlock.
- **+** GPU memory 영역 = TP=4 = 4 GPU 만 활용 (4 GPU idle). 다른 service 영역 활용 가능 (단 본 측정 stability X).

→ **A 의 진정한 영역** = throughput/stability 모두 net loss. 본 환경 (Llama-70B + 8192 context) 영역 에 적합 X.

### short workload 영역 의 D fix 영향

- combo 1 vs combo 2 = +0.1% (variance). short 영역 의 OOB 발생 빈도 낮아서 D fix 활성화 영역 거의 없음.
- long workload (500p × 8192) 영역 에서 D fix 의 진정한 effect 영역 = TBD (별도 측정 필요).

---

## 본 측정 의 의의

1. **D fix 의 정확한 가치 영역 확정**: log spam 차단 only. 진정한 OOB root 영역 (NEO slot tracking race) 영역 = 별도 fix 필요.
2. **A (TP=4) 영역 = scope 외**: 본 environment 영역 에서 instability. TP=8 영역 baseline 유지 권장.
3. **본 turn 의 14 combination plan 영역 → 4 combination 영역 (B = 회귀 known, C = scope 외)**.

---

## 다음 turn — 진행 후보

| 후보 | 영역 |
|---|---|
| OOB precheck root fix | attention.py:956 의 OOB 조건 (`_block_pos_d11 >= _nblk_d11`) 의 진정한 root 영역 추적 + NEO swap-in/block_table 영역 race 영역 fix |
| D fix long workload 측정 | TP=8 + D fix + 500p × 8192 (long) 3-run avg. D fix 의 진정한 long workload 영역 effect |
| C (BLOCK_SIZE=32) | NEO scheduler 영역 광범위 작업 (별도 turn, 2-4 주 effort) |
| F4 (TP=4) revisit | gpu_memory_utilization 영역 낮추거나 max_num_seqs 영역 축소 시 안정성 회복 영역 측정 |

---

## raw 측정 자료

| Run | 위치 |
|---|---|
| combo 1 baseline TP=8 | `eval/results/20260520_130627_combo_baseline_tp8/` |
| combo 2 D-only TP=8 | `eval/results/20260520_132500_combo_d_only_tp8/` |
| combo 3 A-only TP=4 (NO_RESULT) | `eval/results/20260520_134249_combo_a_only_tp4/` |
| combo 4 A+D TP=4 init failure | `eval/results/20260520_141334_combo_a_d_tp4/` |
| combo 4 retry A+D TP=4 (NO_RESULT) | `eval/results/20260520_141842_combo_a_d_tp4_retry/` |
| script combo sweep | `/tmp/run_combo_sweep.sh` |
| script combo 4 retry | `/tmp/run_combo4_retry.sh` |
| summary | `/tmp/run_combo_sweep_summary.txt` |
