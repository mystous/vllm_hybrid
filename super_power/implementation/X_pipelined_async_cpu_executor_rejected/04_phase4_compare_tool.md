# X · Phase 4 — Sync vs Async 비교 자동화 Tool

작성일: 2026-04-22 (KST)
작성자: Claude
관련:
- [`01_design_and_plan.md`](01_design_and_plan.md) §4.4 Phase 4
- [`03_phase2_3_impl_and_verification.md`](03_phase2_3_impl_and_verification.md) Phase 2+3 구현 + 검증

---

## 0. 목적

Phase 2+3 (180345) 에서 X 가 **작동함**은 확정. 이제 **얼마나 이득인가** 를 측정. Phase 4 는 sync vs async 를 같은 workload 로 연속 실행 후 **duration / throughput / correctness** 자동 비교.

run_all.sh 의 phase3 snapshot 은 bench 완주 전 강제 종료하므로 성능 숫자 측정 불가 → **별도 script** (`run_compare.sh`) 신설.

---

## 1. 구현 — `run_compare.sh`

위치: `eval/diagnostics/b2_cpu_parallel/run_compare.sh`

### 1.1 동작 흐름
```
┌────────────────────────────────────────────────┐
│ 1. ENV_SRC 복사 → /tmp/run_compare_sync.env    │
│    HYBRID_CPU_ASYNC_EXECUTOR=0 강제 set        │
│ 2. 이전 서버 모두 kill + sleep                  │
│ 3. serve.sh 기동, ready 대기                    │
│ 4. bench.sh 실행, 완주까지 대기 (timeout 2h)    │
│ 5. server kill, 결과 (hybrid.json) 복사         │
└────────────────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────┐
│ 6. 동일 흐름, HYBRID_CPU_ASYNC_EXECUTOR=1       │
└────────────────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────┐
│ 7. 두 hybrid.json 을 읽어 비교 Python script    │
│    → COMPARE_REPORT.md                          │
└────────────────────────────────────────────────┘
```

### 1.2 옵션
| Option | 설명 | Default |
|---|---|---|
| `--env PATH` | 사용할 env file | `g0_h100x8_qwen32b_light_trace.env` |
| `--port N` | 서버 port | 8000 |
| `--ready-timeout N` | 서버 ready 대기 초 | 1200 |
| `--bench-timeout N` | bench 최대 실행 초 | 7200 (2h) |
| `--skip-sync` | sync skip, async 만 | — |
| `--skip-async` | async skip, sync 만 | — |

### 1.3 결과 디렉토리
```
results/compare_<ts>/
├── sync/
│   ├── server_boot.log
│   ├── bench.log
│   ├── hybrid.json
│   ├── env_used.env
│   └── bench_wall_seconds.txt
├── async/
│   └── (동일)
└── COMPARE_REPORT.md
```

---

## 2. COMPARE_REPORT 형식

자동 생성되는 report:

```markdown
# X Phase 4/5 — Sync vs Async 비교

## 핵심 metric

| 항목 | sync | async | Δ |
|---|---:|---:|---:|
| completed            | 8 | 8 | 0% |
| total_output_tokens  | 4096 | 4096 | 0% |
| duration (s)         | 120.3 | 95.1 | -20.9% |
| request_throughput   | 0.067 | 0.084 | +25.4% |
| output_throughput    | 34.1  | 43.0  | +26.1% |
| mean_ttft_ms         | ...  | ... |
| p99_tpot_ms          | ...  | ... |

## 판정
- ✅ async 가 sync 대비 20.9% 빠름 (duration) — X pipeline 의미 있는 이득
```

판정 로직:
- 두 run 의 `completed` 가 다르면 correctness ⚠
- duration 감소율 > 10% → ✅ meaningful
- 0~10% → 🟡 marginal (Phase 1 보수 추정 범위)
- < 0% → ❌ overlap 이 thread overhead 미만

---

## 3. 사용 예시

### Light workload (권장 first test, ~5분 × 2 = 10분)
```bash
bash eval/diagnostics/b2_cpu_parallel/run_compare.sh
```

### Heavy workload (1~2시간 × 2 — 실행 시간 주의)
```bash
bash eval/diagnostics/b2_cpu_parallel/run_compare.sh \
    --env eval/diagnostics/b2_cpu_parallel/g0_h100x8_qwen32b_longctx_trace.env \
    --bench-timeout 3600
```

### async 만 재측정
```bash
bash eval/diagnostics/b2_cpu_parallel/run_compare.sh --skip-sync
```

---

## 4. Phase 4/5 가 cover 하는 것

| 목표 (170451 §4.4/§4.5) | 이 tool 이 측정? |
|---|---|
| Light workload 완주 | ✓ sync + async 각각 |
| Output token 수 비교 | ✓ `total_output_tokens` |
| Correctness (completed 일치) | ✓ `completed` 비교 |
| Deadlock / panic 없음 | ✓ bench rc 와 bench_wall 로 확인 |
| Bench duration 비교 | ✓ duration 직접 비교 |
| Master 점유율 비교 | ✗ (phase3 가 필요, run_all.sh 영역) |
| Worker util 비교 | ✗ (phase3 필요) |

즉 이 tool 은 **성능 숫자 축** 전담. 코어 활용 heatmap 같은 추가 관측은 `run_all.sh` 에서 병행.

---

## 5. 한계

### 5.1 Phase 3 snapshot 미포함
Flame graph 수집 안 함. "숫자만 깔끔" vs "세부 stack 까지". 필요하면 run_all.sh 로 개별 실행 가능.

### 5.2 Correctness 는 "합계 일치" 수준
`completed` 수, `total_output_tokens` 가 같음 = broad correctness.
Bit-equal token sequence 는 현재 측정 안 함 (deterministic sampler + 동일 seed 필요, 별도 작업).

### 5.3 Heavy workload 의 시간
Heavy (16K prefill) 에서는 CPU 가 느려서 1~2 시간 × 2 = 3~4시간 소요 예상. bench_timeout 을 실제 환경에 맞춰 조정 필요.

---

## 6. 다음 단계

1. **먼저 light workload 로 실행** (10분, 빠른 첫 결과)
2. 결과가 기대 (async 가 5%+ 빠름) 대로면 heavy 로 재측정
3. async 가 sync 보다 느리면 MultiProc 으로 pivot 검토 (170451 §6 대안)

### 실행 명령
```bash
rm -rf eval/diagnostics/b2_cpu_parallel/results/compare_*/
pkill -9 -f 'api_server|serve\.sh|CPU_EngineCore|GPU_EngineCore|benchmark_serving' 2>/dev/null; sleep 3
git pull
bash eval/diagnostics/b2_cpu_parallel/run_compare.sh \
    --env eval/diagnostics/b2_cpu_parallel/g0_h100x8_qwen32b_light_trace.env
git add eval/diagnostics/b2_cpu_parallel/results/
git commit -m "X Phase 4: sync vs async 비교"
git push
```

---

## 7. 코드

주요 구성:
- Bash arg parsing (4개 option)
- `run_one` 함수 — sync/async 각각 한 번 실행 (env flag 교체, 서버 기동, bench 완주, 결과 수집)
- Python inline script — 두 hybrid.json 을 읽어 비교 테이블 생성, 판정 로직

syntax check 완료. Python 파싱은 `json.load` 간단 호출, 안전.
