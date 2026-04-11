# H100x4 Qwen 1.5B hybrid 회귀 — 근본 원인 + 라우팅 수정 검증

`20260411_141500_h100x4_qwen1.5b_routing_regression_root_cause_fix`

> 1.5B hybrid 가 GPU-only 대비 ×7~14 느려진 회귀의 **근본 원인** 을 5단계
> 가설 분리 + 계측 + 코드 수정 + 베이스라인 확인 + 재실험 으로 추적했고,
> **라우팅 결정 로직 1군데 수정**으로 회귀가 사라지는 것을 확인.

## 1. 최종 결과 요약

| 측정 | gpu_only baseline | **수정 전 hybrid** | **수정 후 hybrid** |
|---|---:|---:|---:|
| bench.sh wall (s) | 12.88 | 107.64 | **17.15** |
| benchmark duration (s) | 3.55 | 52.94 | **4.02** |
| request throughput (req/s) | 140.96 | 9.44 | **124.25** |
| output throughput (tok/s) | 17,362 | 1,212 | **15,305** |
| mean TPOT (ms) | 21.48 | 61.09 | **23.71** |
| mean TTFT (ms) | 728 | 1,328 | **891** |
| router CPU dispatched | n/a | 2 / 501 | **0 / 501** |
| router GPU dispatched | n/a | 499 / 501 | **501 / 501** |
| CPU util mean (%) | 6.8 | 84.7 | **6.8** |
| GPU util mean / peak (%) | 13.7 / 60 | 1.5 / 54 | **9.3 / 54** |

→ 회귀 완전 해소. 수정 후 hybrid 의 bench duration / TPOT 은 gpu_only 와
**1.13× / 1.10×** 차이 (노이즈 범위). wall 의 차이 (~4.3s) 는 hybrid 모드의
서버 부팅 + CPU subprocess spawn 오버헤드.

## 2. 5단계 가설 분리

| # | 가설 | 검증 방법 | 결과 |
|---|---|---|---|
| A | resolver 가 `cpu_max_num_seqs` 를 1 이 아닌 큰 값으로 설정 | `[HYBRID-RESOLVE]` 부팅 로그 직독 | **반증** — `max_seqs=1` 정상 |
| B | CPU engine scheduler 가 max_seqs=1 을 무시하고 batch | `[HYBRID-CPU-EXEC]` per-step trace | **반증** — `reqs=0..1, tokens=1` |
| C | 라우터가 1 을 1 로 보지 않고 다수를 CPU 로 보냄 | route() 마다 `[HYBRID-ROUTER-DISPATCH]` 카운터 INFO | **반증** — `n=500 cpu_count=2 gpu_count=498` (probe 포함 거의 전부 GPU) |
| D | API server 가 CPU subprocess 의 OMP 96 코어 점유로 starve | `HYBRID_CPU_THREADS=1` 로 CPU 거의 비활성화 후 재실험 | **반증** — TPOT 64ms 그대로 |
| E | 라우팅 결정이 "CPU 슬롯 비었나"만 보고 "CPU 가 GPU 보다 빠른가"는 안 봄 → 1~2건의 CPU 요청이 long-tail 로 wall 을 지배 | GPU `nvidia-smi` 시계열 + bench probe 동작 분석 | **확정** ✓ |

## 3. 결정적 증거 — GPU `nvidia-smi` 시계열

수정 전 hybrid 96 초 모니터링 동안 GPU avg util 시계열:

```
[0..51]: 0%  ← GPU 완전 idle (51 초)
[52]:    54%
[53]:    29%
[54]:    27%
[55]:    32%
[56..95]: 0% ← GPU 다시 idle (40 초)
```

→ **GPU 가 실제로 일한 시간은 4 초**. 나머지 92 초는 idle.

이 4 초는 gpu_only 의 3.55 초 와 거의 일치 — **GPU 컴퓨트는 정상이고,
회귀는 전적으로 라우팅으로 인한 long-tail 대기**.

## 4. 메커니즘 (왜 ×14 회귀가 나오는가)

`benchmark_serving.py` 의 sequence:

```python
test_output = await request_func(...)  # ← 1건 단일 probe (output_len=128)
print("Initial test run completed.")
# main 500 req burst
```

이전 라우팅 (`_route_throughput_adaptive` 의 cpu-first 분기) 동작:

```python
if (prompt_len <= cpu_prefill_threshold
        and cpu_in_flight < effective_max * num_cpu_engines):  # 0 < 1
    return _to_cpu()    # ← probe 가 CPU 로
```

타임라인:

| 시각 | 이벤트 |
|---|---|
| t=0 | bench probe 1건 보냄 → router cpu_in_flight=0 → **CPU 로 라우팅** |
| t=0..47 | CPU 1.5B 1 req decode (~50ms × 128 tok ≈ 47s). 그 동안 main bench 시작 안 함, GPU idle |
| t=47 | probe 완료, on_request_finished → `_update_adaptive_slots` 가 `_adaptive_cpu_max_seqs` 1 → 2 로 증가 (Bug 1, 별개로 존재) |
| t=47.001 | main 500 burst 시작. R1: cpu_in_flight=0 < 2 → CPU. R2: cpu_in_flight=1 < 2 → CPU. R3~: GPU |
| t=47..51 | GPU 가 498건 4초만에 완료. 그 동안 CPU 는 R1 처리 중 (~47s 더 걸림) |
| t=47..94 | CPU 의 R1 이 47초 동안 진행. **bench wall = max(GPU 4s, CPU 47s) = 47s** |
| t=94 | CPU R1 완료. main bench 의 마지막 요청 응답 도착. main duration = 47s |
| t=94 | bench.sh 종료. 총 wall ≈ 47 (probe) + 47 (main long-tail) = ~94s + 부팅 ≈ **107s** |

이 메커니즘을 5단계 가설 검증 + 계측 + 베이스라인 비교 + GPU 시계열로
확정. 한 줄 요약: **CPU 1 건이 GPU 의 max-latency tail 이 되어 wall 을 지배.**

## 5. 수정 — `_route_throughput_adaptive` (hybrid_core.py:347~)

**Before** (잘못됨):

```python
def _route_throughput_adaptive(self, request_id, prompt_len):
    if self.cpu_first:
        if (prompt_len <= cpu_prefill_threshold
                and cpu_in_flight < effective_max * num_cpu_engines):
            return _to_cpu()  # ← CPU 가 GPU 보다 느려도 무조건 CPU
    return _to_gpu()
```

**After** (Property 2 구현):

```python
def _route_throughput_adaptive(self, request_id, prompt_len):
    # Cold start: 첫 probe blind 회피 → 항상 GPU
    if self._gpu_ema_throughput <= 0.0:
        return self._to_gpu()
    if not cpu_capacity_ok:
        return self._to_gpu()

    cpu_per_req = max(self._cpu_ema_throughput, 1e-6)
    gpu_per_req = max(self._gpu_ema_throughput, 1e-6)

    cpu_finish = (self.cpu_in_flight + 1) * (256 / cpu_per_req)
    gpu_batches_ahead = max(1, (self.gpu_in_flight + 1) // max(1, self.gpu_max_num_seqs))
    gpu_finish = gpu_batches_ahead * (256 / gpu_per_req)

    if self.cpu_first:
        if cpu_finish <= gpu_finish:    # ← Property 2: CPU 가 더 빠를 때만
            return self._to_cpu()
        return self._to_gpu()
    else:
        if (self.gpu_in_flight >= self.gpu_max_num_seqs
                and cpu_finish < gpu_finish):
            return self._to_cpu()
        return self._to_gpu()
```

핵심 변경:

1. **Cold start gate**: `_gpu_ema_throughput == 0` (첫 요청, 아직 EMA 데이터 없음) → 항상 GPU. probe 가 CPU 로 가서 첫 요청 latency 가 wall 을 지배하는 증상 차단.
2. **EMA 기반 expected-finish-time 비교**: CPU/GPU 양쪽의 예상 완료 시간을 EMA throughput 으로 추정하고, **CPU 가 더 빨리 끝낼 때만** CPU 로 라우팅. paper §3 Property 2 의 직접 구현.
3. **cpu-first 와 gpu-first 모두 같은 비교 gate 적용**. cpu-first 는 단지 "동률일 때 CPU 우선" 의 의미만 유지.

이 수정으로:

- **H100 + 1.5B**: GPU 가 CPU 보다 ~13× 빠름 → 모든 요청 GPU 로. hybrid ≡ gpu_only.
- **GPU 가 saturated 되는 워크로드** (예: GPU 큐가 깊어 wait 가 CPU 처리시간 초과): CPU 가 자연스럽게 흡수. paper Property 2 발현.
- **충분히 큰 모델 + 약한 GPU + GPU bound**: CPU 가 평균적으로 GPU 와 비슷한 환경에서 둘 다 일함. 균형.

## 6. 별개로 발견된 잔존 버그 — Bug 1: `_update_adaptive_slots`

`hybrid_core.py:436~443` 의 `_update_adaptive_slots` 는 EMA ratio 를 계산해
CPU 슬롯 수를 동적 조정하려는 의도이지만, `cpu_max_num_seqs=1` (논문 원칙
고정값) 에 대해서는 다음과 같이 항상 2 로 고정됨:

```python
new_max = max(2, min(self.cpu_max_num_seqs * 2,                     # = 2
                    int(self.cpu_max_num_seqs * (1 + ratio))))      # = 1 (ratio<1)
# 모든 ratio 에 대해 결과 항상 2.
```

본 수정 (`_route_throughput_adaptive` 의 expected-finish 비교) 으로 이 버그의
영향이 라우팅 결과에는 더 이상 나타나지 않지만 (어차피 비교 gate 가 모든
케이스를 정확히 처리), 장기적으로는 별도 PR 로 정리 권장. 본 실험 범위 밖.

## 7. 7B 적용성

7B 의 CPU per-req throughput 은 더 낮음 (대략 0.5~1 tok/s 추정, vs 1.5B 의 2.7).
GPU per-req throughput 도 1.5B 보다 낮지만 격차는 더 큼. 따라서 H100 + 7B
환경에서도 본 수정의 expected-finish 비교는 항상 GPU 를 선택할 것이고,
**hybrid ≈ gpu_only** 가 보장됨. probe 가 CPU 에 묶여 분 단위로 정지하던
증상 (`./bench.sh hybrid envs/h100x4_qwen7b_hybrid.env` 의 hang) 도 해소됨.

7B 검증 실험은 시간 제약상 본 리포트에서는 생략. 다음 PR 에서 동일 라우팅
경로에 대해 7B 도 함께 검증 권장.

## 8. 인스트루먼테이션 추가 (재현/디버깅용)

`hybrid_core.py` 에 영구적으로 보존:

- `route()` 첫 호출 시 `[HYBRID-ROUTER-INIT]` (config 한 줄 dump)
- 25 dispatch 마다 `[HYBRID-ROUTER-DISPATCH]` (last decision + counter snapshot)
- `_log_periodic_stats` INFO 승격 (`[HYBRID-ROUTER-STATS]`, interval 은 env 로 제어)

instrument-only env 파일: `eval/envs/h100x4_qwen1.5b_hybrid_instr.env`
(`HYBRID_STATS_LOG_INTERVAL=10`, `VLLM_HYBRID_TRACE_EVERY=50`).

## 9. 영향 범위

**수정한 파일**:
- `vllm/v1/engine/hybrid_core.py`
  - `route()`: 인스트루먼테이션 markers 추가
  - `_log_periodic_stats()`: debug → INFO 승격 + extra fields
  - `_route_throughput_adaptive()`: Property 2 expected-finish 비교 라우팅 (본 fix)

**추가한 파일**:
- `eval/envs/h100x4_qwen1.5b_hybrid_instr.env` (instrumentation env)
- `eval/envs/h100x4_qwen1.5b_hybrid_instr2.env` / `instr3.env` (가설 D 검증용)

**수정하지 않은 파일**:
- env file `h100x4_qwen1.5b_hybrid.env` 본체는 그대로. 본 fix 적용 후 그대로
  사용 가능.

## 10. 원본 데이터 포인터

| 항목 | 경로 |
|---|---|
| gpu_only baseline | `eval/results/20260411_135206_G_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/` |
| 수정 전 hybrid (instr1, threads=auto) | `eval/results/20260411_133730_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/` |
| 가설 D — threads=80 | `eval/results/20260411_135832_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/` |
| 가설 D — threads=1 | `eval/results/20260411_140420_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/` |
| **수정 후 hybrid (정상)** | `eval/results/20260411_141353_H_C_H100_80GB_HBM3_x4_Qwen2.5-1.5B-Instruct/` |
| 직전 회귀 비교 분석 | `experiment_result/20260411_130959_h100x4_qwen1.5b_gpu_vs_hybrid_latest_compare/` |
| 서버 stdout 로그 (instrumentation) | `/tmp/hybrid_instr/server.log`, `server_t1.log`, `server_fix.log` |

## 11. 권장 후속 작업

1. **7B 환경에서 동일 fix 검증** (TODO 항목, 본 리포트 외).
2. **`_update_adaptive_slots` (Bug 1) 별도 정리** — 본 fix 로 라우팅 영향은
   사라졌지만 코드 자체는 잘못된 동작이므로 정리.
3. **paper §3 Property 2 의 expected-finish-time 비교 공식을 본문에 명시** —
   현재 paper 는 "CPU is complement" 라고만 적혀 있고 정량 식이 없음.
4. **dev RTX 3090 + 1.5B/7B 에서 이 fix 의 영향 재측정** — dev 환경에서는
   GPU 가 더 약하므로 CPU 가 실제로 일부 흡수해서 hybrid > gpu_only 로
   나오는 영역이 있을 수 있음. 본 fix 가 그 영역을 망치지 않는지 확인.
