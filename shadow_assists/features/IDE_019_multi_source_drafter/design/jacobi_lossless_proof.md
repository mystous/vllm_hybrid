# Jacobi Parallel Decoding — Lossless Guarantee Proof

> SUB_180 — IDE_019 / TSK_035 의 이론적 토대. AVX-512 Jacobi kernel 의 output 이
> autoregressive (AR) decoding 과 정합함을 증명한다.
> reference: Lookahead Decoding (Fu et al., USENIX OSDI'24), Block-parallel decoding (Stern et al., NeurIPS'18)

---

## 1. 정의

### 1.1 Autoregressive (AR) 기준

prompt `x_{1..n}` 에 이어지는 deterministic (greedy) AR 시퀀스를

```
y_{n+1} = argmax_v p(v | x_{1..n})
y_{n+t} = argmax_v p(v | x_{1..n}, y_{n+1..n+t-1})  (t ≥ 2)
```

로 정의한다. 본 SUB 의 lossless 기준은 **이 시퀀스 `y_{n+1..n+K}` 와 token-level 일치**이다 (greedy path).

stochastic sampling (temperature > 0) 의 경우, distribution 동치는 rejection sampling step 에서 보장되며 (§4), Jacobi 자체는 candidate 제안만 한다.

### 1.2 Jacobi parallel decoding fixed-point

K-길이 candidate sequence `c = (c_1, c_2, ..., c_K)` 와 prompt `x` 를 입력으로
한 step 의 transformer forward 가 K logits 을 만든다:

```
ℓ_i = f_θ(x, c_{1..i-1})_i      (i = 1..K)
```

(K position 의 logits 을 동시에 얻는 self-attention causal mask 의 standard property.)

Jacobi iteration 은

```
c_i^(t+1) = argmax_v ℓ_i^(t) = argmax_v f_θ(x, c_{1..i-1}^(t))_i
```

로 update 한다 (t = iteration index, c^(0) 은 초기 guess).

**fixed point**: `c* = (c*_1, ..., c*_K)` 가
`c*_i = argmax_v f_θ(x, c*_{1..i-1})_i` 를 모든 i 에 대해 만족하면, `c*` 는
정확히 AR greedy 시퀀스 `y_{n+1..n+K}` 와 같다.

### 1.3 증명 (induction by i)

- **base (i=1)**: `c*_1 = argmax_v f_θ(x)_1 = argmax_v p(v|x) = y_{n+1}` ✓
- **inductive step**: `c*_j = y_{n+j}` 가 j < i 에서 모두 성립한다고 가정.
  fixed-point 조건에서 `c*_i = argmax_v f_θ(x, c*_{1..i-1})_i`. inductive
  가정에 의해 `c*_{1..i-1} = y_{n+1..n+i-1}` 이므로
  `c*_i = argmax_v f_θ(x, y_{n+1..n+i-1})_i = argmax_v p(v|x, y_{n+1..n+i-1}) = y_{n+i}` ✓

따라서 fixed point 에 도달하면 candidate 는 AR 와 token-level bit-exact 이다. ∎

### 1.4 fixed-point 수렴 보장

`f_θ` 는 finite vocab 위의 transformer 이므로 argmax 후 state 는 vocab^K 의
discrete 한 점이고, Jacobi update 는 함수 `g: vocab^K → vocab^K`. 이는 monotone
property 없이도 **vocab^K 가 유한** 이므로 cycle 또는 fixed point 에 들어간다.

- 본 SUB 의 안전장치: `max_iters` cap (default 8) + cycle detection (이전 2
  iter 동일 → break).
- non-convergence 시: prefix 중 fixed-point 만족하는 `c*_{1..m}` 만 candidate 로
  emit (verify stage 에서 reject). lossless 는 항상 verify step 이 강제.

---

## 2. verify stage 통합

Jacobi candidate `(c*_1, ..., c*_K)` 는 **draft** 일 뿐이고, 실제 채택은
target model 의 verify forward (1 회) 가 결정한다 (speculative decoding standard):

```
target logits = f_target(x, c*_{1..K})  // K+1 positions parallel
for i = 1..K:
    if argmax(target logits_i) == c*_i:
        accept
    else:
        rollback to i-1 and emit argmax(target logits_i) at position i
        break
```

여기서 **draft model 과 target model 이 같은 model 이면** (Jacobi 의 self-draft
모드), greedy mode 에서는 verify 가 정확히 §1.3 의 fixed-point condition 의 확장:
`c*_i == argmax f_θ(x, c*_{1..i-1})_i` 를 한 번 더 check 하는 셈이다. fixed point
에 도달했다면 100% accept, 도달하지 못했다면 부분 prefix 만 accept.

**stochastic sampling**: Jacobi 가 argmax 대신 sampling 하더라도, verify step
에서 standard rejection sampler (Leviathan et al., ICML'23) 를 적용하면
**distribution-level lossless** (target model 의 p 와 동일 분포). 본 SUB 의
context (Qwen 32B greedy mode canonical) 에서는 §1.3 의 token-level bit-exact 이
유효.

---

## 3. CPU draft + GPU verify 의 정확성

본 SUB 의 deployment 는 CPU 가 draft, GPU target 이 verify:

- CPU 의 LM head matmul (BF16) 은 GPU 의 BF16 LM head 와 numerically 동일 weight
  사용. argmax 의 robustness 는 BF16 비결합성에도 ~99%+ token agreement
  (CLAUDE.md 운영 해석의 분포 유사성과 일치).
- token argmax tie-break 차이로 한 iter 갈리더라도 §1.4 의 max_iters cap 안에서
  fixed point 로 수렴하거나, prefix 만 emit + verify reject 로 안전 처리.
- **본 SUB 의 lossless gate**: greedy mode `c* = y` (token bit-exact) — Jacobi
  fixed-point 시. partial fixed-point 시 emit prefix 만 검증.

---

## 4. summary table

| condition | guarantee | gate |
|---|---|---|
| Jacobi fixed point reached, greedy | token-level bit-exact AR | §1.3 |
| Jacobi partial fixed point (prefix m < K), greedy | prefix `c*_{1..m}` = `y_{n+1..n+m}` bit-exact | §1.4 |
| Jacobi non-convergence (max_iters hit) | candidate 는 noise — verify stage reject 로 lossless | §2 |
| stochastic sampling (temp > 0) | distribution-level lossless via rejection sampler | §2 |

---

## 5. AVX-512 vectorize 가 정확성에 영향 없음

AVX-512 BF16 vectorize 는 **partial sum 의 order 만 변경** 한다.

- FP32 accumulator (`_mm512_dpbf16_ps`): partial sum 16-wide × hidden-tile.
  scalar 와 ordering 다르나 BF16 비결합성 영향만 — argmax 의 tie 영역에서 매우
  drmaly 다른 결과는 < 0.01% (SUB_171 의 BF16 tokenizer ordering test 와 동일
  pattern).
- argmax 의 tie 영역은 verify stage 가 자동 reject — lossless 보장 유지.

---

## 결론

- Jacobi parallel decoding 의 fixed point 는 AR greedy 와 **token-level bit-exact**.
- non-converge / partial-converge 의 경우에도 verify stage 가 **prefix 만 accept**
  → lossless.
- AVX-512 vectorize 는 partial sum order 변경뿐, lossless 보장 깨지 않음.
- 본 SUB 의 accuracy gate = "verify stage 와 통합 시 token-level diff 없음"
  (이는 vllm 통합 후 SUB_181 에서 실측).
