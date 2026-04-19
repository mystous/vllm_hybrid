# 06. Hot Path 연결 증명 (G1 진입 필수)

**Tier**: 1
**상태**: 🔶 **Phase A 구현 완료 (2026-04-19, 측정 대기)** — Q8_0 dispatch 경로로 MLP (gate_up_proj + down_proj) hot path 치환. H100 측정 전
**중요도**: **G0 계측 다음 본선 진입의 첫 관문**

---

## Phase A 구현 요약 (2026-04-19)

- **Scheme**: llama.cpp Q8_0 (weight INT8 + fp16 per-block scale, block=32). Activation 은 Python 쪽 BF16/FP32 유지 — kernel 내부에서 per-row dynamic INT8 양자화 → VNNI dot → FP32 accumulate → dtype-matched output. WoQ 로 분류. §04 IPEX WoQ 기각 후 대체 경로로 §23 CPU Native Quantization 과 자연스럽게 통합
- **Scope — strict whitelist**:
  - **Arch**: `Qwen2ForCausalLM` 만 허용. Qwen2Moe / Qwen2_5_VL / Qwen2Audio / Qwen3 / LLaMA 계열은 MLP 구조 미검증 상태라 **자동 skip**
  - **Module**: `*.mlp.gate_up_proj` + `*.mlp.down_proj` 만 매칭. 아래 substring 하나라도 포함되면 제외 — `experts.`, `vision`, `visual`, `speech`, `audio` (MoE expert / 멀티모달 타워 차단)
- **IPEX 관계 (명확히)**:
  - **패치된 layer 만** apply-time 에 IPEX / oneDNN 을 우회 — `quant_method.apply()` 자체가 교체되므로 IPEX 가 설치한 optimized module 경로로 돌아가지 않음
  - **패치되지 않은 나머지 layer** (attention, norm, embedding, lm_head 등) 는 IPEX 가 이미 설치한 최적화 경로를 그대로 사용. 즉 Phase A 는 "MLP 만 vLLM-hybrid native, 나머지는 IPEX" 구조
  - patch 호출은 `load_model` 의 **IPEX optimize 이후 + LoRA load 이후** 시점 → LoRA delta 가 base weight 에 반영된 뒤 1회 quantize
- **LoRA 비호환**: static Q8_0 quantized qweight 는 runtime delta-W adapter swap 과 충돌 → `lora_config` 가 있으면 `patch_mlp_to_q8_0` 이 조기 return + warning. Adapter hot-swap 지원은 Phase B 이후
- **Load-time quantize**: `torch.ops._C_cpu_ops.q8_0_quantize_weight` 로 load 직후 1회 변환. runtime repack 없음 (`[HYBRID-KERNEL]` summary log 에 `repack=0` 명시)
- **활성 경로 (기존 hybrid-* 설정과 동일 패턴)**: env 파일의 `HYBRID_VNNI_HOT_PATH=1` → `serve.sh` 가 `--hybrid-vnni-hot-path` CLI arg 로 변환 → `HybridConfig.vnni_hot_path=True` → `patch_mlp_to_q8_0` 이 `hybrid_config.vnni_hot_path` 를 읽고 활성. `os.getenv` 직접 참조 없음 (설정값은 CLI, 관측용만 env export 하는 프로젝트 관례 준수)
- **Guard 5겹**: `hybrid_config.vnni_hot_path=True` + `HAS_CPU_OPS` + op registered + arch ∈ allowlist + LoRA 미사용. 실패 시 warning + no-op
- **Layer-aware trace**: `VLLM_HYBRID_KERNEL_TRACE=1` 시 `[HYBRID-KERNEL-Q8_0] layer=<full_qualified_name> M=... N=... K=... time=...ms` per-call 로그. 부팅 요약 log 에 `patched=N skipped=M arch=... lora=...` 기록

**신규/수정 파일**:
- 신규: `vllm/v1/worker/hot_path_wiring.py`
- 수정: `vllm/v1/worker/cpu_model_runner.py` (load_model 끝, **LoRA load 이후** 지점 hook)

**dev smoke test 통과** (2026-04-19, RTX3090 + i9-12900KF AVX2):
- import OK, `_cpu_ops_available = False` 정확히 감지 → no-op + warning
- qweight 크기 공식: Qwen2.5-32B TP=8 기준 gate_up_proj [6912, 5120] → 35.9 MB (BF16 대비 0.51× 메모리)

---

## Phase A 실행 방법 (H100x8)

**1. 사전 확인 — q8_0 op 빌드 여부 (재빌드 필요 여부 판단)**
```bash
python -c "import torch, vllm._C_cpu_ops; print('q8_0=', hasattr(torch.ops._C_cpu_ops,'q8_0_linear'), 'quantize=', hasattr(torch.ops._C_cpu_ops,'q8_0_quantize_weight'))"
```
둘 다 `True` 면 재빌드 불필요 (Python 변경뿐). 하나라도 `False` 면:
```bash
pip install -e . --config-settings="cmake.args=-DVLLM_TARGET_DEVICE=cuda"
```

**2. 측정 (g0_06 sweep seqs 1/4/16)**
```bash
cp eval/envs/g0_h100x8_qwen32b_06.env /tmp/run.env
for s in 1 4 16; do sed -i "s/^HYBRID_CPU_MAX_SEQS=.*/HYBRID_CPU_MAX_SEQS=$s/" /tmp/run.env; ./eval/serve.sh hybrid /tmp/run.env & SERVE_PID=$!; until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 5; done; ./eval/bench.sh hybrid /tmp/run.env; kill $SERVE_PID; wait $SERVE_PID 2>/dev/null; mkdir -p measurement_results/H100x8/g0_06/seqs$s; mv eval/results/$(ls -t eval/results/ | head -1) measurement_results/H100x8/g0_06/seqs$s/; done
```

**3. 비교 대상 (baseline 이미 측정됨)**
- `measurement_results/H100x8/g0_00_32b/seqs{1,4,16}/` — §06 미적용 (동일 env, `HYBRID_VNNI_HOT_PATH=0`)
- `measurement_results/H100x8/g0_00_32b/gpu_only_baseline/` — wall ratio 기준

**4. G1 통과 판정 3축**
| 지표 | 계산 | 통과 조건 |
|---|---|---|
| Batch scaling | `per_req_cost(seqs=4) / per_req_cost(seqs=1)` | ≤ 2.0 |
| Tail | GPU bulk 완료 후 CPU drain 시간 | < 100 s |
| Wall ratio | `hybrid wall / gpu_only wall` (동일 workload) | < 8× |

**5. 성공 시 확인할 log 마커 (`hybrid_server_boot.log`)**
```
[HYBRID-KERNEL] §06 patched=128 skipped=0 (filter=0, error=0) arch=Qwen2ForCausalLM lora=False scope=Qwen2_MLP(.mlp.gate_up_proj,.mlp.down_proj) quantize=load-time-1x repack=0 non_patched_layers=ipex_unchanged
```
`patched` = 64 layer × 2 proj = **128** 여야 정상 (Qwen2.5-32B 의 MLP 커버리지).

**6. 실패 시 (Stop/Go Case 3)**
`seqs=1` 만 빨라지고 `seqs=4` scaling 없으면 hot path 이득은 single-req 최적화로만 귀속. 다음 Tier (§07~) 진입 금지. kernel-level marker (`VLLM_HYBRID_KERNEL_TRACE=1` + 요청 수 축소) 로 병목 재추적.

---

## 왜 필요한가

현재 vLLM hybrid 의 CPU 경로는 IPEX `ipex.llm.optimize` 에 의존. IPEX 는 내부적으로 oneDNN primitive 를 호출하며, 어떤 primitive 가 선택되는지는 IPEX/oneDNN dispatcher 가 결정. 우리가 빌드한 `_C_cpu_ops.abi3.so` 의 VNNI INT8 GEMM (`csrc/cpu/gemm_vnni.cpp`) 은 **build 는 되지만 실제 Python 쪽에서 호출하는 경로 (call-site) 가 없음**.

**Codex 원칙**:
> "VNNI 가 있다" 도 신규 gain 아님 — hot path 에 실제 연결되어 batch scaling 을 만든 경우만 gain.

이 섹션은 **Tier 2/3 kernel 실험의 전제**. hot path 를 실제로 바꾸는 infra 없이는 §07~§17 의 kernel 개선도 "IPEX 가 계속 사용됨" 으로 귀결.

---

## 기술적 배경

### 현재 CPU linear path

```
Python model forward
  → vllm.model_executor.layers.linear.Linear.forward
  → (ipex.llm.optimize 에 의해 치환) IPEX optimized Linear
  → IPEX MLP/Attn fused primitive
  → oneDNN primitive (IPEX dispatcher 선택)
  → AMX BF16 matmul (SPR 자동) 또는 AVX-512 VNNI INT8 (quant 경로)
```

우리가 구현한 `int8_gemm_vnni` (`csrc/cpu/gemm_vnni.cpp`) 는 이 chain 어디에도 **callable 연결 안 됨**.

### 연결 지점 후보

**Option A — IPEX bypass**:
- `vllm.model_executor.layers.linear.Linear.forward` 를 IPEX 치환 **전에** 우리 kernel 로 override
- 장점: 명시적 제어
- 단점: IPEX 의 다른 최적화 (pre-pack, kernel fusion) 잃음

**Option B — IPEX hook**:
- IPEX 의 `_IPEXLinearFusionCPU` 혹은 oneDNN custom primitive 로 등록
- 장점: IPEX 최적화 유지
- 단점: IPEX 내부 구조 의존, 버전별 깨짐

**Option C — Torch custom op**:
- `torch.library.define("vllm_cpu::gemm_vnni", ...)` 로 등록
- model load 후 `post_process_model_hook` 으로 특정 Linear 를 torch.ops 호출로 치환
- 장점: Python 에서 명시 dispatch, IPEX 와 공존
- 단점: hook 설계 필요

**권장**: Option C (torch custom op + post-load hook). `csrc/cpu/torch_bindings_hybrid.cpp` 가 이미 일부 op (VNNI) 를 torch ops 로 등록했을 가능성 — 확인 필요.

### Pre-pack cache

AMX tile layout (16×64 bytes) 또는 VNNI block layout (16×4 bytes) 로 weight 를 load-time 에 1회 재배치. runtime repack 을 방지:
- IPEX 가 이미 `weights_prepack=True` 암묵 활성
- **그러나 우리 kernel (VNNI INT8) 은 IPEX 의 pre-pack format 과 다를 수 있음** — 별도 pre-pack 필요

### Shape 별 dispatch log

decode 시 `(M=batch, N=hidden_dim, K=intermediate)` shape 이 매 step 거의 동일. prefill 은 M=prompt_len. 우리가 VNNI INT8 에 대해 어느 shape 에서 언제 호출됐는지 로그:
```
[HYBRID-KERNEL] vnni_int8 M=16 N=4096 K=11008 time=2.3ms
[HYBRID-KERNEL] ipex_amx_bf16 M=1 N=4096 K=4096 time=0.8ms
[HYBRID-KERNEL] sdpa_fallback M=1 N=4096 K=128 time=1.1ms
```

`ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI` 같은 설정이 있다고 해서 실제 VNNI 가 호출된다는 보장은 없음. dispatcher 결정은 shape / alignment / pre-pack format 에 따름.

---

## 관련 참고 문헌

- **Codex playbook §6 Tier 1 Mainline Hot Path 연결**: `/vllm_hybrid/ideation/20260415_094148_codex_ninja_gap_modification_playbook.md`
- **KTransformers AMX**: https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md — AMX-specialized kernel 을 실제 hot path 에 연결하는 방식
- **KTransformers SOSP'25 paper**: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf — CPU/GPU hybrid 에서 synchronization overhead 감소 방식
- **SGLang + KTransformers blog**: https://lmsys.org/blog/2025-10-22-KTransformers/ — ISA switching + coordination overhead 감소
- **oneDNN Primitive Dispatch**: https://oneapi-src.github.io/oneDNN/
- **Torch custom op**: https://pytorch.org/tutorials/advanced/torch_library.html
- **Tech_done v1 Q1**: `_C_cpu_ops` 빌드 검증 기록

---

## 구체 작업

- [ ] **torch_bindings_hybrid.cpp 검토**: 현재 어떤 op 가 torch ops 로 등록되어 있는지 확인 (`torch.ops._C_cpu_ops.xxx` 로 callable 한 것 나열)
- [ ] **VNNI INT8 GEMM op 등록 완비**: `torch.ops._C_cpu_ops.int8_gemm_vnni(..)` callable 확인, schema 문서화
- [ ] **Linear module patch 설계**: `vllm_patch_linear_to_vnni(model)` — 특정 layer (`o_proj`, `down_proj` 등 memory-bound 선별) 를 torch custom op 호출로 치환
- [ ] **Load-time pre-pack cache**: `cpu_model_runner.py` 의 `load_model` 후 hook 에서 weight 를 VNNI layout 으로 1회 변환 + serialized tensor 로 cache
  - `~/.cache/vllm_hybrid/prepack/<model_hash>_vnni.pt`
- [ ] **Runtime repack 계측**: step 마다 weight layout 전환이 일어나는지 marker `[HYBRID-KERNEL-REPACK]` 기록 (0 이어야 함)
- [ ] **Shape 별 dispatch log marker**: `[HYBRID-KERNEL] shape=... kernel=... time=...`
  - env variable `VLLM_HYBRID_KERNEL_TRACE=1` 일 때만 출력 (overhead 방지)
- [ ] **`ONEDNN_MAX_CPU_ISA` 실제 dispatch 확인**: oneDNN verbose `ONEDNN_VERBOSE=1` 와 교차 검증, primitive 이름 (`brg_matmul:avx512_core_vnni` vs `brg_matmul:avx10_1_512_amx`) 기록
- [ ] **성공 판정 실험**: `num_seqs=1/2/4/8` 각각에서 VNNI kernel hit rate + per-req cost

---

## 성공 조건

1. ✅ `[HYBRID-KERNEL]` 로그에서 VNNI INT8 kernel 이 실제 호출됨을 확인
2. ✅ `ONEDNN_VERBOSE` 가 우리 custom op 의 호출을 보여줌 (또는 직접 kernel 이면 IPEX 우회 확인)
3. ✅ runtime repack count = 0 (pre-pack cache 작동)
4. ✅ `num_seqs=4` 에서 per-request cost 감소 (G0 baseline 대비)
5. ✅ **단일 req 만 빨라지고 scaling 없으면 다음 Tier 금지** (Stop/Go Case 3)

---

## 의존성

- **선행**: §01 G0 계측 (baseline + sublayer breakdown 확보), §05 OMP env
- **병행**: §03 Huge Pages (pre-pack cache 효율에 유리)
- **후속**: §07 ISA dispatch 가 본 infra 를 사용해 batch shape 별 분기. §15 AMX pre-pack 은 본 pre-pack cache 확장
- **ultimately**: §10, §11, §13, §14 모두 hot path 연결 이후에만 의미

---

## 리스크

- **IPEX 가 내부적으로 이미 VNNI 쓰고 있음**: 우리 kernel 이 더 빠르다고 보장 없음. KTransformers 수치 (1.5–2×) 는 IPEX 부재 조건
- **pre-pack format 불일치**: IPEX weights_prepack 과 우리 VNNI pre-pack 이 다르면 모델 weight 를 두 벌 보관 (메모리 2×)
- **Torch custom op 의 fx graph / compile 친화성**: vLLM 의 model compile 경로 (`torch.compile`) 와 충돌 가능. graph break 발생 시 성능 영향 측정
- **dispatch 분기 overhead**: 매 layer 마다 shape check → Python overhead. `torch.ops` 는 빠르지만 layer 수 80+ 면 누적 가능. 측정

---

## 스택 호환성

- §07 ISA Binary Dispatch: 본 infra 위에 batch 조건 기반 분기 추가
- §13 T-MAC LUT GEMV: 본 infra 를 모델로 삼아 INT4 LUT op 등록
- §14 AVX/AMX Cascade: 본 infra + tile buffer 관리
- §15 AMX Pre-pack: 본 pre-pack cache 의 AMX tile layout 확장

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| env `HYBRID_VNNI_HOT_PATH` | `0` (기본) / `1` | serve.sh 가 `--hybrid-vnni-hot-path` CLI arg 로 변환, `HybridConfig.vnni_hot_path` 에 저장 |
| `VLLM_HYBRID_KERNEL_TRACE` | `0` (기본) / `1` | per-call Q8_0 kernel 호출 trace (observability, serve.sh 에서 명시 export) |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `csrc/cpu/gemm_vnni.cpp` — VNNI INT8 GEMM 구현
- `csrc/cpu/torch_bindings_hybrid.cpp` — torch ops 등록 (확인/보강)
- `vllm/v1/worker/cpu_model_runner.py` — load_model 후 hook 위치
- `vllm/_custom_ops.py` — `HAS_CPU_OPS` flag + callable 노출
- `vllm/model_executor/layers/linear.py` — Linear module patch 대상
- `vllm/v1/attention/backends/cpu_attn.py` — `_decode_path_counts` (기존 counter)
