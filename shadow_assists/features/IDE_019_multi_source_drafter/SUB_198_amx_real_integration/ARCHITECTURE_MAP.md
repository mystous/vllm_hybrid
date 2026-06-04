# SUB_198 — Qwen2.5-0.5B-Instruct CPU Draft Architecture Map

> **parent**: SUB_198 (real spec-decode integration) — produced as part of
> SUB_201 A1 lever real-forward scaffold turn.
>
> **purpose**: map each Qwen2.5-0.5B-Instruct decoder operation to (a) the
> existing AMX kernel coverage in `SUB_187/src/amx_draft_qwen05b.cpp` and
> (b) the PyTorch CPU draft scaffold in `vllm/v1/spec_decode/cpu_amx.py`
> (this turn). Identifies the per-op gap that real AMX integration must
> close, with rough sub-task estimates.

---

## 1. Qwen2.5-0.5B-Instruct — model config

Source: `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/.../config.json`

| field | value |
|---|---|
| architectures | `Qwen2ForCausalLM` |
| hidden_size | **896** |
| intermediate_size | **4864** |
| num_hidden_layers | **24** |
| num_attention_heads | **14** (head_dim = 64) |
| num_key_value_heads | **2** (GQA, kv_groups = 7) |
| vocab_size | **151936** |
| max_position_embeddings | 32768 |
| tie_word_embeddings | **true** (embed_tokens.weight == lm_head.weight) |
| rope_theta | 1_000_000.0 |
| dtype | bfloat16 |

Per-layer ops (Qwen2 decoder, prefill 1 token / decode):
1. `input_layernorm` — RMSNorm over hidden=896
2. `self_attn`:
   - `q_proj` (896 → 896 = 14×64) BF16 GEMM
   - `k_proj` (896 → 128 = 2×64) BF16 GEMM
   - `v_proj` (896 → 128 = 2×64) BF16 GEMM
   - RoPE on q, k (head_dim=64)
   - GQA dot-product attention (q[14, 64] vs k_cache[2, S, 64] with kv_groups=7)
   - softmax over S
   - attn × v_cache[2, S, 64]
   - `o_proj` (896 → 896) BF16 GEMM
3. `post_attention_layernorm` — RMSNorm
4. `mlp` (SwiGLU):
   - `gate_proj` (896 → 4864) BF16 GEMM + SiLU
   - `up_proj` (896 → 4864) BF16 GEMM
   - elementwise multiply
   - `down_proj` (4864 → 896) BF16 GEMM
5. residual add

After 24 layers: `model.norm` (final RMSNorm) → `lm_head` (896 → 151936) BF16
GEMM (weight tied to embedding).

---

## 2. AMX kernel coverage — `SUB_187/src/amx_draft_qwen05b.cpp`

| op | shape | covered? | notes |
|---|---|---|---|
| LM-head GEMM | (B,896) × (896,151936) | **YES** | `amx_matmul_bf16_omp_n` + repack, used in `amx_draft_qwen05b_step_ms`. Verified 1.44 ms K=7 (microbench, B=1, dev box). |
| MLP-gate GEMM | (B,896) × (896,4864) | **YES** | `amx_draft_qwen05b_mlp_ms`. Only `gate_proj` weight allocated — `up_proj`, `down_proj` not modeled. |
| Token embedding lookup | id → (896,) | **NO** | Trivial gather (no AMX needed) but missing in kernel. |
| RMSNorm | (B,896) | **NO** | AVX-512 BF16 reduction needed (cheap, ~µs). |
| Attention q_proj / k_proj / v_proj / o_proj | (B,896) × (896, [896 or 128]) | **NO** | Same AMX BF16 GEMM recipe as LM-head but with different N. k_proj/v_proj have N=128 < tile_N=16 multiple OK. |
| RoPE | (B, heads, 64) | **NO** | Elementwise + sin/cos table. No AMX. AVX-512 sufficient. |
| GQA attention (Q·K^T, softmax, A·V) | (B, 14, 64) · (B, 2, S, 64) | **NO** | Decode mode is a row × matrix × matrix chain; tile shape needs careful mapping (M=14 < tile_M=16). |
| KV cache write | append (k,v) per layer | **NO** | No CPU KV cache structure exists in kernel today. |
| RMSNorm × 49 (24 input + 24 post + 1 final) | trivial | **NO** | Same as above. |
| residual add | (B,896) | **NO** | AVX-512 trivial. |

**Coverage summary**: AMX kernel today covers **LM-head GEMM** (dominant
cost in vocab path) and **one MLP gate GEMM**. It is a microbench, not a
forward pass — there is no attention, no KV cache, no norm, no token
embedding, no actual layer chain. The 24-layer "linear chain" in the
microbench is `K` *repeats of the same matmul on the same buffer*, not 24
distinct layer ops.

---

## 3. AMX integration — 4 sub-task estimate (real forward)

To turn the microbench into a real Qwen-0.5B BF16 forward producer
suitable for spec-decode draft, four follow-on sub-tasks are required.

| sub-task | scope | est | notes |
|---|---|---:|---|
| **(a) attention** | q/k/v/o proj GEMMs + RoPE + GQA dot + softmax + KV cache layout (AMX tile aligned: tile_M=16, tile_K=32 BF16, tile_N=16) | **~5-7 dev-days** | Decode mode M=1 wasteful for tile_M=16; consider tile_M=14 partial or accept padding. K=64 (head_dim) too small for K=32 inner — fuse two heads or use AVX-512 fallback for attention dot. |
| **(b) token embedding + final norm** | id-gather + RMSNorm fused | **~1 dev-day** | Tied weights — reuse LM-head weight buffer. AVX-512 sufficient. |
| **(c) RMSNorm + SwiGLU + residual** | AVX-512 BF16 elementwise + reduction | **~1-2 dev-days** | 49 RMSNorms + 24 SwiGLU + 24 residuals. Cheap individually but must be fused with GEMM output to avoid memory bandwidth bottleneck. |
| **(d) integration** | 24-layer chain driver, weight loader from HF safetensors → repacked BF16, ctypes ABI, sampler (greedy + temperature), tokenizer bridge, vllm spec_decode wire-up validation | **~5-7 dev-days** | Includes accuracy gate (compare to PyTorch CPU reference on per-token logprob max-abs-diff per CLAUDE.md §Constraint). |

**Total**: ~12-17 dev-days of focused work. Matches the SUB_198
scaffold's "2-3 weeks" upstream estimate.

---

## 4. PyTorch CPU draft scaffold (this turn)

Implementation: `vllm/v1/spec_decode/cpu_amx.py` — extended from
89-line toy to a **PyTorch CPU forward** path gated by env vars.

| component | implementation |
|---|---|
| model load | `transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=bf16)`, CPU only, lazy-loaded on first `propose()` |
| thread pool | `torch.set_num_threads(VLLM_CPU_DRAFT_THREADS or 8)`; AMX kernels auto-used by oneDNN/MKL when BF16 + bf16 ISA available |
| draft loop | K iterative greedy steps; each step: forward the growing token tensor (no KV cache yet), `argmax(logits[-1])`, append |
| trigger | env `VLLM_USE_AMX_DRAFT=1` enables real forward; otherwise toy fallback (deterministic `last+1..last+K`) so dispatch wire-up still PoC-bootable |
| safety | model load failure / transformers missing → silent fallback to toy + warning print once |
| smoke harness | `tests/test_cpu_amx_smoke.py` — instantiates proposer with mocked vllm_config, calls `propose([[785,3974,13876,38835]])`, asserts (a) returns list of ints, (b) ids are in vocab range |

Result: **~180 lines** added to `cpu_amx.py`, model loads OK, propose()
returns real next-token ids. **AMX kernel itself is still not called from
this path** — this scaffold uses PyTorch's CPU GEMM (which on Sapphire
Rapids will auto-dispatch to AMX via oneDNN bf16 path, but on the dev
Alder Lake box uses AVX-512 BF16 only).

---

## 5. Two paths forward

### Path A — ship PyTorch CPU draft to vllm integration (fast)
- ~33 ms/step on dev box (8 threads, Alder Lake AVX-512 BF16, no AMX)
- Sapphire Rapids estimate: ~10-15 ms/step (AMX auto-dispatch via oneDNN)
- K=7 draft ≈ 70-100 ms — already above the ~40 ms GPU verify budget
- **Verdict**: PyTorch CPU path is too slow for net-positive spec-decode.
  Useful as functional baseline only (verify pipeline gets *real* drafts,
  acceptance rate measurable). Net throughput still negative until AMX
  custom kernel integration lands.

### Path B — finish AMX integration (12-17 dev-days)
- Target: <5 ms/step → K=7 ≈ 35 ms (parity with GPU verify, paper §4 target)
- Pre-requisite: sub-tasks (a)-(d) in §3
- Risk: KV cache mgmt for paged-KV vllm interop, accuracy gate (per
  CLAUDE.md §Constraint — per-token logprob max-abs-diff, not bit-exact)

**Recommended dev step (next turn)**: Path A wire-up validation (boot
vllm with `--speculative-config '{"method":"cpu_amx_draft","num_speculative_tokens":7}'`,
confirm draft tokens flow into verify pipeline, measure acceptance rate),
then begin Path B sub-task (a) attention kernel.
