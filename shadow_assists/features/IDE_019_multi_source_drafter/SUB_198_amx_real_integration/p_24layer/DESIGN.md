# DESIGN — Qwen2.5-0.5B 24-layer AMX CPU forward (Phase A1 / SUB_198)

> **parent**: SUB_198 Phase A3-real (lm_head 단일 matmul 통합 — top-1 100%
> 8/8, logprob max-abs-diff 0.0077) 의 후속.
> **scope**: 단일 matmul → **진짜 Qwen-0.5B 24-layer forward** (attention
> + RMSNorm + SwiGLU + RoPE + KV cache + lm_head) 의 AMX CPU 구현 설계.
> **deliverable**: 본 turn 의 design (이 파일) + C++ 구현 (`src/amx_draft_qwen05b.cpp`
> 확장) + 단일 layer 검증 + 24-layer chain 검증 + microbench.
> **host**: Intel Xeon Platinum 8570 (Sapphire Rapids 후속, AMX BF16 native).

---

## 1. 두괄식

| 항목 | 값 |
|---|---|
| target model | Qwen2.5-0.5B-Instruct (HF cache, BF16 safetensors) |
| layers | 24 (Transformer block, decoder-only) |
| hidden | 896 |
| Q heads | 14 (head_dim = 64) |
| KV heads | 2 (GQA group = 7) |
| intermediate | 4864 (SwiGLU MLP) |
| vocab | 151,936 (config) / 152,064 (kernel padded) |
| RoPE θ | 1,000,000 |
| RMSNorm ε | 1e-6 |
| tie embeddings | true (lm_head = embed_tokens) |
| dtype 통일 | BF16 (storage) / FP32 (matmul 누산) |
| target latency (full forward B=1) | 25 ms/step 목표 (GPU verify 40 ms 와 net positive) |

---

## 2. Transformer block 분해

각 decoder layer 는 다음 sub-step 으로 구성:

```mermaid
flowchart TB
    H[hidden h_in:(B,S,896) BF16] --> N1[RMSNorm input_layernorm w:896]
    N1 --> Q[q_proj 896×896 + bias]
    N1 --> K[k_proj 896×128 + bias]
    N1 --> V[v_proj 896×128 + bias]
    Q --> RQ[RoPE θ=1e6 per head_dim=64]
    K --> RK[RoPE θ=1e6 per head_dim=64]
    RQ --> AT[GQA attn 14Q × 2KV, scale=1/sqrt 64]
    RK --> KV[KV cache append]
    V --> KV
    KV --> AT
    AT --> O[o_proj 896×896]
    O --> R1[residual add: h_in + o_proj_out]
    R1 --> N2[RMSNorm post_attention_layernorm w:896]
    N2 --> G[gate_proj 896×4864]
    N2 --> U[up_proj 896×4864]
    G --> SI[SiLU sigmoid·x]
    SI --> MUL[× up_proj_out]
    U --> MUL
    MUL --> D[down_proj 4864×896]
    D --> R2[residual add: R1 + down_proj_out → h_out]
```

### 2.1 sub-op cost 분석 (per layer, B=1, S=1, decode-step)

| sub-op | shape | FLOPs | 누적 cost (24L) | AMX OK? |
|---|---|---:|---:|---|
| RMSNorm × 2 | (1,896) | 2×896 mul + reduce | ~85 K | AVX-512 (BF16→FP32) |
| q_proj | (1,896)×(896,896) | 1.6 MFLOP | 38.6 MFLOP | **AMX BF16** |
| k_proj | (1,896)×(896,128) | 229 KFLOP | 5.5 MFLOP | **AMX BF16** |
| v_proj | (1,896)×(896,128) | 229 KFLOP | 5.5 MFLOP | **AMX BF16** |
| RoPE × Q+K | 14·64 + 2·64 dims | <50 K | <1.2 MFLOP | scalar/AVX |
| GQA attn (S_kv=1) | Q·K^T + softmax + ·V | 14·64·1 ≈ 1 K + 256 KFLOP | <8 MFLOP | scalar/AVX (small) |
| o_proj | (1,896)×(896,896) | 1.6 MFLOP | 38.6 MFLOP | **AMX BF16** |
| gate_proj | (1,896)×(896,4864) | 8.7 MFLOP | 209 MFLOP | **AMX BF16** |
| up_proj | (1,896)×(896,4864) | 8.7 MFLOP | 209 MFLOP | **AMX BF16** |
| down_proj | (1,4864)×(4864,896) | 8.7 MFLOP | 209 MFLOP | **AMX BF16** |
| SwiGLU σ·× | (1,4864) | 14.6 K | 350 K | AVX |
| **per-layer total** | — | ≈ 29.6 MFLOP | — | — |
| **24-layer total** | — | — | ≈ 711 MFLOP | — |
| **lm_head** | (1,896)×(896,152064) | 272 MFLOP | — | **AMX BF16** ✅ (이미 통합) |
| **grand total** | — | — | **≈ 983 MFLOP/step** | — |

→ **8480+ AMX peak 22 TFLOPS BF16** 기준 이론 lower bound: 983e6 / 22e12 = **45 μs**.
   실측은 cache miss + per-call overhead 로 SUB_187 lm_head 단독 0.97 ms (8480+, OMP=16).
   24 layer 의 cumulative real-world latency 목표 **25 ms/step**.

### 2.2 KV cache layout

Decode-only forward (S=1 추론) 가정. KV cache 는 layer 별로 prefix sequence 의 K,V 를 누적.
본 PoC 단일 forward smoke test 는 **prefix 길이 = S_prompt** (예: 8 토큰) 의 prefill 한 번
+ 1 토큰 decode 까지. KV cache 는 in-memory `std::vector` (eviction 없음).

KV layout (per layer):
- `K_cache[layer][seq=S][KV_HEAD=2, HEAD_DIM=64]` BF16, row-major
- `V_cache[layer][seq=S][KV_HEAD=2, HEAD_DIM=64]` BF16, row-major

GQA attention 시 14 Q head 를 2 KV head 로 mapping (group_size=7):
- q_head_idx 0..6 → kv_head_idx 0
- q_head_idx 7..13 → kv_head_idx 1

---

## 3. weight key mapping (safetensors → AMX kernel)

본 kernel 의 `qwen05b_layer_t` 구조에 다음 key 를 BF16 그대로 load 후 AMX-packing.

| safetensors key | shape | kernel field | repack |
|---|---|---|---|
| `model.embed_tokens.weight` | [151936, 896] | `embed_tokens` | row-major BF16 (token_id row lookup) |
| `model.layers.{L}.input_layernorm.weight` | [896] | `layers[L].ln1_w` | BF16 그대로 |
| `model.layers.{L}.self_attn.q_proj.weight` | [896, 896] | `layers[L].q_w_packed` | transpose→AMX pack [H/2, N, 2] |
| `model.layers.{L}.self_attn.q_proj.bias` | [896] | `layers[L].q_b` | BF16 그대로 |
| `model.layers.{L}.self_attn.k_proj.weight` | [128, 896] | `layers[L].k_w_packed` | transpose→AMX pack |
| `model.layers.{L}.self_attn.k_proj.bias` | [128] | `layers[L].k_b` | BF16 그대로 |
| `model.layers.{L}.self_attn.v_proj.weight` | [128, 896] | `layers[L].v_w_packed` | transpose→AMX pack |
| `model.layers.{L}.self_attn.v_proj.bias` | [128] | `layers[L].v_b` | BF16 그대로 |
| `model.layers.{L}.self_attn.o_proj.weight` | [896, 896] | `layers[L].o_w_packed` | transpose→AMX pack |
| `model.layers.{L}.post_attention_layernorm.weight` | [896] | `layers[L].ln2_w` | BF16 그대로 |
| `model.layers.{L}.mlp.gate_proj.weight` | [4864, 896] | `layers[L].gate_w_packed` | transpose→AMX pack |
| `model.layers.{L}.mlp.up_proj.weight` | [4864, 896] | `layers[L].up_w_packed` | transpose→AMX pack |
| `model.layers.{L}.mlp.down_proj.weight` | [896, 4864] | `layers[L].down_w_packed` | transpose→AMX pack |
| `model.norm.weight` | [896] | `final_norm_w` | BF16 그대로 (lm_head 직전) |
| (tied) `model.embed_tokens.weight` | [151936, 896] | `lm_head_packed` | zero-pad to 152064 + transpose→AMX pack (Phase A3-real 이미 완료) |

### 3.1 GQA 차원 주의

- `q_proj` output 차원 = 896 = num_attention_heads × head_dim = 14 × 64
- `k_proj`/`v_proj` output 차원 = 128 = num_key_value_heads × head_dim = 2 × 64
- `o_proj` input/output 차원 = 896

→ AMX matmul 의 **N 차원이 16 의 배수여야 함**. 본 kernel 에서:
   - 896 = 16 × 56 ✓
   - 128 = 16 × 8 ✓ (k_proj/v_proj OK)
   - 4864 = 16 × 304 ✓
   - 152064 = 16 × 9504 ✓

### 3.2 K 차원 (matmul reduction dim) 주의

- AMX 의 `_tile_dpbf16ps` 는 BF16 pair (k_pair = 2) 단위로 누산. **K % 32 == 0** 필요.
- 모든 K 차원 (896, 4864) 은 32 의 배수 ✓.

---

## 4. AMX register / tile layout

본 kernel 의 single thread tile config (SUB_187 에서 검증된 layout 재사용):

| tile | role | shape | bytes/row | rows |
|---|---|---|---:|---:|
| TMM0 | C (accumulator FP32) | 16×16 | 64 | 16 |
| TMM1 | A (BF16) | 16×32 BF16 | 64 | 16 |
| TMM2 | B (BF16 pair-packed) | 16 K-pair × (16 BF16 pair) | 64 | 16 |

→ 모든 weight matmul 은 **M=16 (B_amx)**, N=tile 단위 16, K=32-pair-loop 으로 분해.
   B=1 의 decode-only 는 M=1 → AMX tile constraint 위해 M=16 round-up
   (위 16 rows 중 1 row 만 유효, 나머지는 zeros 또는 dummy. logits 도 첫 row 만 사용).

---

## 5. OMP parallelism 전략

| sub-op | parallelism axis | thread count |
|---|---|---:|
| q_proj / o_proj | over N tiles (896/16 = 56 tiles) | OMP_NUM_THREADS (16) |
| k_proj / v_proj | over N tiles (128/16 = 8 tiles) | 최대 8 |
| gate_proj / up_proj | over N tiles (4864/16 = 304 tiles) | OMP_NUM_THREADS (16) |
| down_proj | over N tiles (896/16 = 56 tiles) | OMP_NUM_THREADS (16) |
| lm_head | over N tiles (152064/16 = 9504 tiles) | OMP_NUM_THREADS (16) |
| RMSNorm | scalar / AVX-512 inner loop | single thread (work too small) |
| RoPE | per-head loop (14+2 = 16 heads) | optional OMP, 본 PoC 는 single thread |
| Softmax / attn matmul | per-head loop | single thread (S=1, 비용 무시 가능) |
| SwiGLU activation | elementwise 4864 dim | AVX-512 single thread |

→ matmul 외 op 는 모두 single thread (decode-step latency 가 너무 작아 thread spawn 비용 > work).

---

## 6. RoPE 구현

Qwen2 RoPE: θ_i = θ^(-2i/d) where d = head_dim = 64, θ = 1e6.

per position `p`, per head_dim pair `(2i, 2i+1)`:
- `q_new[2i]   = q[2i]   * cos(p·θ_i) - q[2i+1] * sin(p·θ_i)`
- `q_new[2i+1] = q[2i]   * sin(p·θ_i) + q[2i+1] * cos(p·θ_i)`

BF16 precision: cos/sin table 을 FP32 로 precompute (max_pos × head_dim/2),
runtime 에서 BF16 q,k 를 FP32 로 cast → rotate → BF16 cast back.

본 PoC max_pos = 64 (smoke test 짧은 prompt) — table 크기 = 64 × 32 × 2 (cos,sin) FP32 = 16 KB.

---

## 7. RMSNorm 구현

`y = (x / sqrt(mean(x^2) + ε)) * w`

BF16 → FP32 cast → variance reduce → rsqrt → scale → BF16 cast back.

AVX-512: 896 dim = 56 × 16 lanes (BF16). 단일 thread 충분.

---

## 8. Forward API 설계

새 C 함수 추가 (기존 `amx_draft_qwen05b_forward` 와 별개):

```c
// Single-layer forward (P2 validation)
int amx_draft_qwen05b_layer_forward(
    int layer_idx,                // 0..23
    const uint16_t* h_in,         // [S, HIDDEN(896)] BF16
    int S,                        // 1..MAX_SEQ
    uint16_t* h_out);             // [S, HIDDEN(896)] BF16

// Full 24-layer forward + lm_head (P3)
int amx_draft_qwen05b_forward_full(
    const int32_t* input_ids,     // [S] int32 token ids
    int S,                        // prompt length
    int32_t* out_ids,             // [K] sampled tokens (greedy argmax)
    int K,                        // generation steps
    uint16_t* logits_last_bf16);  // [K, VOCAB_PADDED(152064)] BF16, last-step logits per K

// Weight load (call once after init)
int amx_draft_qwen05b_load_layer_weights(
    int layer_idx,
    const uint16_t* ln1_w,   // [896]
    const uint16_t* q_w,     // [896, 896] row-major BF16
    const uint16_t* q_b,     // [896]
    const uint16_t* k_w,     // [128, 896]
    const uint16_t* k_b,     // [128]
    const uint16_t* v_w,     // [128, 896]
    const uint16_t* v_b,     // [128]
    const uint16_t* o_w,     // [896, 896]
    const uint16_t* ln2_w,   // [896]
    const uint16_t* gate_w,  // [4864, 896]
    const uint16_t* up_w,    // [4864, 896]
    const uint16_t* down_w); // [896, 4864]

int amx_draft_qwen05b_load_embed_tokens(
    const uint16_t* embed,    // [151936, 896]
    int vocab_valid,
    int hidden);

int amx_draft_qwen05b_load_final_norm(
    const uint16_t* w);       // [896]
```

---

## 9. 검증 게이트 (CLAUDE.md §Constraint 운영해석)

| Phase | gate | threshold |
|---|---|---|
| P2 single-layer | layer[0] AMX output vs HF transformers reference layer[0] output | per-element max-abs-diff (BF16) < 0.05 OR cosine sim > 0.999 |
| P3 24-layer + lm_head | next-token top-1 match | ≥ 90% (5 prompt) |
| P3 24-layer + lm_head | next-token top-3 match | ≥ 95% |
| P3 logprob distributional | per-token logprob max-abs-diff | < 0.1 |
| P4 microbench | per-step full-forward latency B=1 K=7 | < 40 ms (GPU verify budget) → PASS gate |

---

## 10. 본 turn 산출 / 다음 turn 산출 경계

| 항목 | 본 turn (24-layer chain) | 후속 |
|---|---|---|
| DESIGN.md | ✓ (이 파일) | — |
| C++ kernel layer ops | ✓ (RMSNorm, RoPE, attn, MLP, SwiGLU) | — |
| weight load API | ✓ (`load_layer_weights`, `load_embed_tokens`, `load_final_norm`) | — |
| single-layer test | ✓ (P2) | — |
| 24-layer + lm_head test | ✓ (P3, smoke 5+ prompt) | — |
| latency microbench | ✓ (P4, B=1 K=7) | — |
| 모델 확장 (Qwen-7B, Llama-7B) | — | 후속 SUB |
| vLLM e2e wire-up | — | 별도 invasive SUB |

---

## 11. 알려진 위험

- **AMX K=32 / M=16 / N=16 alignment**: k_proj/v_proj output 128 → OK. 단 batch dim M=1 → 16 round-up 으로 15 rows wasted.
- **prefill 의 attention complexity**: S 가 커지면 attn = O(S²·H). 본 PoC 는 S ≤ 16 short prompt 만.
- **PyTorch reference 일치성**: BF16 matmul 의 reduction order 차이로 token argmax 가 한 번 갈리면 cascading divergence (CLAUDE.md §Constraint 운영해석). 따라서 binding metric = top-1/top-3 + logprob max-abs-diff.
- **SiLU 정밀도**: `x * sigmoid(x)` 에서 sigmoid 의 exp 정확도가 BF16 한계. FP32 cast 후 활성화 → BF16 cast back.

---

## 12. 참조

- SUB_187 `src/amx_draft_qwen05b.cpp` — AMX BF16 matmul kernel 기반.
- SUB_198 `AMX_INTEGRATION_DESIGN.md` — lm_head 통합 (Phase A3-real).
- CLAUDE.md §Constraint — 정확도 게이트 운영해석 (분포 유사성).
- HF transformers `modeling_qwen2.py` — reference forward 구현.
