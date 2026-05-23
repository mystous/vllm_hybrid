# SUB_033 — B3 FlashDecoding++ unified-max / online softmax 적재 plan

> **parent**: TSK_019 / 영역 B B3 ([N 문서 #4.2](../analysis/N_cdec_leftover_elimination_ideas.md))
> **출처**: 사용자 명시 (turn 18) — "권고 사항에 맞춰 진행해" → O 분석 §7.2 ★★★ 첫 항목.
> **현 status**: plan 단계 (코드 변경 전).

---

## 1. 배경 — 왜 softmax 단축이 가장 큰 lever 인가

O 분석 §6.3 의 다음 lever 표:

| 항목 | 정체 | 효과 가설 |
|---|---|---|
| **B3** | **FlashDecoding++ unified-max softmax** | softmax recomputation 제거 → cdec kernel 호출 횟수 자체 감소 |
| **B1** | **OmniServe LSE async merge** | LSE merge 를 GPU 와 async → CPU 의 sync wait 제거 |

B3 surface 가 **pacpu.ispc 의 softmax 영역** (외부 영향 없음, ISPC rebuild 만 필요) 으로 가장 작음 → 첫 시도.

## 2. 현 코드 (pacpu.ispc:111-142) — 3-pass softmax

```c
export void softmax(seq_len, softmax_scale, a, asb) {
  // Pass 1: scale + local-max
  foreach (h = 0 ... NUM_Q_HEADS) {
    amb[h] = -1e20;
    for (i = 0; i < seq_len; i++) {
      a[i*NQH + h] *= softmax_scale;        // memory write
      amb[h] = max(amb[h], a[i*NQH + h]);
    }
  }
  // Pass 2: exp + sum
  foreach (h = 0 ... NUM_Q_HEADS) {
    asb[h] = 0;
    for (i = 0; i < seq_len; i++) {
      a[i*NQH + h] = exp(a[i*NQH + h] - amb[h]);  // memory read+write
      asb[h] += a[i*NQH + h];
    }
  }
  // Pass 3: normalize + log-sum-exp
  foreach (h = 0 ... NUM_Q_HEADS) {
    for (i = 0; i < seq_len; i++) {
      a[i*NQH + h] /= asb[h];               // memory read+write
    }
    asb[h] = log(asb[h]) + amb[h];
  }
}
```

**문제**:
- `a[]` 를 **3번 읽고/쓰는** memory traffic (seq_len ≈ 8K × NUM_Q_HEADS).
- Pass 사이에 `foreach (h)` barrier → libgomp barrier wait 누적.
- AVX-512 SPR 의 streamer prefetch 가 잡지만 L2 reuse 가 깨짐 (pass 사이 L2 evict).

## 3. 변경안 — Online softmax (FA2 / FlashDecoding++) 단일 pass + final normalize

**핵심 아이디어**: max 와 sum 을 **online update** 로 한 번에 계산 (FlashAttention2 의 online softmax).

```c
// SUB_033 online softmax — env-gated VLLM_NEO_SOFTMAX_ONLINE=1
export void softmax_online(seq_len, softmax_scale, a, asb) {
  uniform itmd_t amb[NUM_Q_HEADS];
  // Pass 1 (fused): scale + online max + online sum
  foreach (h = 0 ... NUM_Q_HEADS) {
    amb[h] = -1e20;
    asb[h] = 0;
    for (i = 0; i < seq_len; i++) {
      itmd_t s = a[i*NQH + h] * softmax_scale;
      a[i*NQH + h] = s;                     // overwrite (1 write)
      itmd_t new_max = max(amb[h], s);
      asb[h] = asb[h] * exp(amb[h] - new_max) + exp(s - new_max);
      amb[h] = new_max;
    }
  }
  // Pass 2: normalize + log-sum-exp
  foreach (h = 0 ... NUM_Q_HEADS) {
    for (i = 0; i < seq_len; i++) {
      a[i*NQH + h] = exp(a[i*NQH + h] - amb[h]) / asb[h];
    }
    asb[h] = log(asb[h]) + amb[h];
  }
}
```

**효과 가설**:
- Memory traffic: **3 passes → 2 passes** (1 pass 줄임 = 33% 절감).
- `foreach (h)` barrier: **3 → 2** (libgomp barrier #1 감소).
- exp 호출: seq_len → 2*seq_len (+seq_len) — compute 증가.
- net: SPR 가 memory-bound 라면 +, compute-bound 라면 -.

## 4. 정확도 보장 — Constraint 부합 검증

- Online softmax 는 floating-point associativity 가 아니어서 결과가 **bit-exact 동일하지 않음** (CLAUDE.md "GPU만 사용 결과 동일" Constraint 의 *분포·의도 수준* 해석 안).
- 검증 방법:
  1. Per-token logprob max abs diff
  2. 시퀀스 PPL relative diff
- TST_003 (정확도 게이트) 와 동일 base prompts 사용.

## 5. 적재 step

| Step | 작업 | site | effort |
|---|---|---|:-:|
| **5.1** | `csrc/cpu/pacpu/pacpu.ispc` 에 `softmax_online` 함수 추가 + env-gated 분기 | pacpu.ispc | 30 min |
| **5.2** | `core.h` 에 env override 로직 추가 (`VLLM_NEO_SOFTMAX_ONLINE`) | core.h | 15 min |
| 5.3 | pacpu rebuild (`pip install -e . --no-build-isolation`) | build | 5 min |
| 5.4 | 정확도 검증 (TST_003 base prompts) | eval | 30 min |
| 5.5 | 100p × 8192 측정 (A4+B3 ON / A4-only / baseline 3-way) | eval | ~45 min × 3 = 2 hr |
| 5.6 | 3-run avg (winner case) | eval | 90 min |

**총 effort**: ~4-5 hr (단 단계 측정 fail 시 +1-2 hr fall-back).

## 6. risk + fallback

| risk | mitigation |
|---|---|
| ISPC 가 `exp` × seq_len 더 호출하면서 compute-bound 로 회귀 | env=0 fallback 유지, 기존 `softmax` 그대로 보존 |
| 정확도 PPL 차이 > 게이트 (0.5% rel) | env=0 default → opt-in |
| ISPC compiler 가 online loop vectorize 못 함 | foreach (h) lane 단위 unroll 명시 + ispc `-O3` |

## 7. 측정 plan

1. **baseline** (current 3-pass): env-OFF (A4 only), 100p × 8192 → 941.1 추정
2. **B3 ON** (online softmax + A4): env=1, 100p × 8192 → 가설 945~955 tps (+0.5~1.5%)
3. **B3 OFF + A4** (regression check): env=0, 100p × 8192 → baseline 와 동일해야 함 (코드만 추가, default 미사용)

위 3 way 1-run 측정 → 차이 > ±2% noise 시 3-run avg 진입.

## 8. 다음 turn 의 deliverables

1. `csrc/cpu/pacpu/pacpu.ispc` 수정 (`softmax_online` 추가)
2. `csrc/cpu/pacpu/core.h` env-gated branching
3. pacpu rebuild + 정확도 verify log
4. measurement SUMMARY.tsv (3-way A4 / A4+B3_ON / A4+B3_OFF)
5. `measurements/sub033_flashdec_plusplus_<date>/RESULTS.md`
6. id_registry.md SUB_033 상태 갱신 ("활성 → 완료/기각")
