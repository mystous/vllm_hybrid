# SUB_037 — B4 SPARAMX ARI-aware AMX↔AVX-512 switch plan

> **parent**: TSK_019 / N 문서 영역 B B4 / O 분석 §7.2 ★★ B-tier 후보
> **출처**: 사용자 명시 (turn 22) — "3개 전부 다 진행"
> **선행**: SUB_036 (Path A 500p baseline) 결과 — 500p 워크로드 에서 lever signal 가능성 확정 후 진입.

---

## 1. 배경

### 1.1 SPARAMX 논문 (arXiv 2502.12444)

| 항목 | 값 |
|---|---|
| 핵심 | **ARI (Arithmetic Intensity)** 기반 AMX vs AVX-512 dispatch |
| 가정 | low ARI (memory-bound) workload 는 AVX-512 가 빠르고, high ARI (compute-bound) 는 AMX 가 빠르다 |
| NEO 적용 | sequence-length / prefill vs decode / block_size 에 따라 dispatch 분기 — 현재 NEO 는 USE_AMX=1 시 모두 AMX |

### 1.2 P3 회귀 컨텍스트 (NEO 기존 시도)

- P3 (F3) — K BF16 + AMX qk path = 이전 -2.5% 회귀, 현재 net zero
- 회귀 root cause = 일부 sequence-length 영역에서 AMX 의 tile_dpbf16ps 의 setup overhead (`tile_loadd`/`tile_stored`) 가 compute 절감 압도
- → ARI 기반 dispatch 가 정확히 본 회귀 복구

## 2. 현 코드 surface

### 2.1 dispatcher (core.h:405-450)

```cpp
static thread_local int _amx_decided = 0;
if (_amx_decided == 0) {
  const char* env = std::getenv("VLLM_NEO_USE_AMX");
  bool want_amx = (env && env[0] && env[0] != '0');
  if (want_amx && amx_kernel::ensure_amx_init()) {
    _amx_decided = 1;
  } else {
    _amx_decided = -1;
  }
}
// ... per-task loop
if (_amx_decided == 1) {
  if (k_is_bf16) amx_kernel::attn_one_seq_amx_bf16(...);
  else amx_kernel::attn_one_seq_amx(...);
} else {
  ispc::attn_one_seq(...);
}
```

→ **현재 dispatcher 는 모듈 init 1회만 결정** (thread_local flag). per-task / per-seq ARI dispatch 없음.

### 2.2 SPARAMX dispatch 적용안

```cpp
// SUB_037 B4 — per-task ARI dispatch
// ARI(qk) ≈ seq_len * HEAD_DIM * NUM_Q_HEADS / (seq_len * HEAD_DIM + NUM_Q_HEADS * HEAD_DIM)
//         ≈ NUM_Q_HEADS (대형 head, AMX 유리) for seq_len >> NUM_Q_HEADS
// ARI(av) ≈ seq_len * HEAD_DIM / (seq_len * NUM_Q_HEADS + HEAD_DIM)
//         ≈ low for high NUM_Q_HEADS → AVX-512 유리

for (auto t = thrd_start_task[tid]; t < ...; t++) {
  int seg_len = std::get<2>(tasks[t]);

  // SUB_037 — ARI threshold (env-tunable VLLM_NEO_ARI_THRESHOLD_SEQ)
  bool use_amx = (_amx_decided == 1) && (seg_len >= _ari_seq_threshold);

  if (use_amx) {
    amx_kernel::attn_one_seq_amx_bf16(...);
  } else {
    ispc::attn_one_seq(...);  // AVX-512 path
  }
}
```

env: `VLLM_NEO_ARI_THRESHOLD_SEQ` (default 256, sweep 64/128/256/512/1024)

## 3. 적재 step

| Step | 작업 | site | effort |
|---|---|---|:-:|
| 3.1 | env-gated `VLLM_NEO_ARI_THRESHOLD_SEQ` 추가 + per-task `seg_len` 기반 dispatch | `core.h` | 30 min |
| 3.2 | rebuild | build | 5 min |
| 3.3 | 정확도 verify (B-tier 변경 영역 attention 결과 동일) | eval | 30 min |
| 3.4 | threshold sweep (500p × 8192, threshold=64/128/256/512/1024 + baseline AMX-only) | eval | 6 × 25 min = 2.5 hr |
| 3.5 | best threshold 3-run avg (winner) | eval | 75 min |

**총 effort**: ~4-5 hr

## 4. 위험

| risk | mitigation |
|---|---|
| ARI threshold 가 본 워크로드 에서 모두 AMX 가 유리 (= 무효) | threshold=∞ (= 항상 AMX, 현 상태) 와 threshold=0 (= 항상 ISPC) 양 끝 측정으로 sanity check |
| ISPC path 가 BF16 K 미지원 | `if (k_is_bf16)` 분기 + ISPC fallback 시 K_FP16 로 cast — 단 cast 비용 | 
| per-task dispatch 의 branch mispredict 가 overhead | dispatch 결과를 task 그룹별로 sort 후 batch dispatch |

## 5. SUB_036 결과 의존성

| SUB_036 결과 | SUB_037 진입 결정 |
|---|---|
| NEO 500p 가 noise 안 (saturated) | SUB_037 시도 가치 낮음 → SUB_039 (av_amx 확장) 우선 |
| NEO 500p 가 noise 위 (lever signal 검출 가능) | SUB_037 정식 진입 |
| vanilla 500p >> NEO 500p | NEO net benefit 재정의 — 더 큰 워크로드 (vanilla OOM 영역) 로 이동 후 SUB_037 |
