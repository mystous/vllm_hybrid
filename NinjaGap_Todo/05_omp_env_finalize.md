# 05. OMP env 마무리 (KMP_BLOCKTIME)

**Tier**: 0
**상태**: ✅ 거의 완료 (`KMP_BLOCKTIME=0` 만 H100 env 파일에 누락)
**예상 이득**: <5% (90% 이미 적용)

---

## 왜 "거의 완료" 인가

핵심 OMP/NUMA 설정은 이미 C++ `init_cpu_threads_env` (`csrc/cpu/utils.cpp`) 에서 강제됨:

```cpp
// csrc/cpu/utils.cpp:55-95
numa_set_membind(mask);          // NUMA memory strict bind
numa_set_strict(1);              // strict enforcement
#pragma omp parallel for schedule(static, 1)
for (size_t i = 0; i < omp_cpu_ids.size(); ++i) {
    sched_setaffinity(...);       // OMP thread 1:1 core pinning
}
```

그리고 `_setup_cpu_process_env` (`hybrid_core.py`) 가 설정하는 환경 변수:
- `OMP_NUM_THREADS=<NUMA phys core count>`
- `MKL_NUM_THREADS=<동일>`
- `OMP_DYNAMIC=FALSE`
- `MKL_DYNAMIC=FALSE`
- `OMP_WAIT_POLICY=ACTIVE`
- `KMP_AFFINITY=granularity=fine,compact,1,0`
- `KMP_TPAUSE=0`
- `VLLM_CPU_OMP_THREADS_BIND=auto`

**`OMP_PROC_BIND=close` 은 의도적 미설정** — Intel OMP runtime 의 master-thread pin bug 유발하여 오히려 악화. `hybrid_core.py` 가 환경에서 `OMP_PROC_BIND` / `OMP_PLACES` 를 pop 하고 `sched_setaffinity` 로 직접 제어.

### 남은 차이: `KMP_BLOCKTIME`

dev env 파일 (`dev_*.env`) 에는 `KMP_BLOCKTIME=0` 이 있지만 **H100 env 파일에 누락**.

---

## 기술적 배경

### `KMP_BLOCKTIME` 이란

Intel OMP runtime 은 parallel region 이 끝난 후 **thread 를 일정 시간 (기본 200ms) 동안 spin-wait 상태로 유지** 한다. 다음 region 이 바로 오면 sleep wake-up 비용 절감.

- `KMP_BLOCKTIME=infinite` (기본): thread 는 다음 region 까지 spin (CPU 사용량 100%)
- `KMP_BLOCKTIME=200`: 200ms spin 후 sleep
- `KMP_BLOCKTIME=0`: parallel region 끝나면 즉시 sleep

### vLLM decode 시나리오 분석

decode per step 시간 ~25ms (IPEX BF16 batch=1). step 간격 ~5ms (scheduler overhead). 즉:
- `KMP_BLOCKTIME=infinite` → step 끝날 때마다 thread 가 200ms spin. step 간격이 5ms 이므로 **대부분의 thread 가 항상 busy-wait** → CPU 다른 작업에 방해 (특히 hybrid 에서 API server / router 가 같은 CPU 경쟁)
- `KMP_BLOCKTIME=0` → step 끝나면 즉시 sleep. wake-up cost (~5-10us) 가 있지만, hybrid environment 에서 CPU 자원을 API server 에게 양보하는 효과

Hybrid dual-process 구조에서 CPU engine 이 **자기 NUMA 의 물리 core 에만** pin 되어 있고, API server 는 다른 core 에 있지만 IPC ZMQ 통신이 동일 cache line 범위 — spin-wait 가 ZMQ recv 에 간접 방해.

### 왜 `OMP_PROC_BIND=close` 를 안 쓰나

`OMP_PROC_BIND=close` 는 "master thread 와 가까운 place 에 worker thread 배치" 를 지시. Intel OMP 런타임의 master-thread 자체는 **place 첫 slot 에 고정** 되는데, master 의 원래 affinity (cgroup 에서 받은 것) 를 무시하고 `OMP_PLACES[0]` 로 이동. 이로 인해:
- dev 환경 (24 logical CPUs, hybrid 구조): master 가 CPU 0 에 묶여 다른 thread 와 동일 core 공유 → contention
- H100x8 환경: master 가 NUMA 0 끝에 몰려 NUMA 1 engine 이 NUMA 0 메모리 access 유발

해결: `OMP_PROC_BIND` / `OMP_PLACES` 를 **환경변수에서 제거** + C++ `sched_setaffinity` 로 직접 제어. 현재 `_setup_cpu_process_env` 에 다음 있음:
```python
os.environ.pop("OMP_PROC_BIND", None)
os.environ.pop("OMP_PLACES", None)
# cgroup effective affinity 복원
effective = os.sched_getaffinity(1)
os.sched_setaffinity(0, effective)
```

---

## 관련 참고 문헌

- **Intel OpenMP Runtime Library Reference — KMP_BLOCKTIME**: https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/kmp-blocktime.html
- **Intel OpenMP KMP_AFFINITY**: https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-10/openmp-options-and-environment-variables.html
- **이전 수정 커밋**: `vllm/v1/engine/hybrid_core.py` 의 `_setup_cpu_process_env` + `csrc/cpu/utils.cpp` 의 `init_cpu_threads_env`
- **Tech_done v5 F1**: NUMA strict bind 검증 완료 기록

---

## 구체 작업

- [ ] **`KMP_BLOCKTIME=0` 을 H100 env 파일에 추가**:
  - `eval/envs/h100x8_qwen7b_hybrid.env`
  - `eval/envs/h100x8_qwen1.5b_hybrid.env`
  - `eval/envs/h100x8_qwen32b_hybrid.env`
  - 기타 H100x4/x8 hybrid env 전체
- [ ] **`_setup_cpu_process_env` 에 fallback 보강**: `os.environ.setdefault("KMP_BLOCKTIME", "0")` 추가 (env 파일에서 누락 시에도 자동)
- [ ] **로그 확인**: `[HYBRID-CPU-ENV]` 에 `KMP_BLOCKTIME=0` 출력되는지 확인
- [ ] **실측**: 동일 workload 에서 `KMP_BLOCKTIME=0` vs `infinite` 비교 — decode step time + API server latency + `top` 의 CPU 사용률 패턴

---

## 성공 조건

- 모든 H100 env 파일에 `KMP_BLOCKTIME=0` 존재
- `[HYBRID-CPU-ENV]` 로그에서 확인
- `top` 에서 decode 간격 동안 CPU 사용률이 100% spin 이 아니라 periodic drop (sleep/wake) 관찰
- API server ZMQ recv latency (`[HYBRID-CLIENT]` marker interval) 감소 (수치는 측정 후 확정)

---

## 의존성

- **선행**: 없음 (즉시 적용 가능)
- **후속**: 모든 기법의 baseline 에 적용

---

## 리스크

- **Wake-up overhead 가 step time 에 가산**: step 간격이 매우 짧은 (`num_seqs` 많을 때) 경우 오히려 악화 가능. 단 현재 `cpu_max_num_seqs=1` 원칙 하에서는 step 간격 5ms 이므로 안전
- **다른 OMP 기반 라이브러리와의 상호작용**: MKL, IPEX 내부 OMP pool 이 같은 env 를 공유하므로 블락타임 0 이 MKL 의 작은 matmul 성능 깎을 수 있음. 측정으로 확인

---

## 스택 호환성

독립. 모든 후속 기법과 조합 가능.

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_KMP_BLOCKTIME` | `auto` (기본) / `0` / `<n>ms` | `auto` 면 hybrid 가 0 으로 강제 |

전체 flag 테이블: [00_Overview.md](./00_Overview.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `vllm/v1/engine/hybrid_core.py` — `_setup_cpu_process_env`
- `csrc/cpu/utils.cpp` — `init_cpu_threads_env`
- `eval/envs/*.env` — env 파일들
- `eval/envs/dev_*.env` — 이미 `KMP_BLOCKTIME=0` 있음 (참조 모델)
