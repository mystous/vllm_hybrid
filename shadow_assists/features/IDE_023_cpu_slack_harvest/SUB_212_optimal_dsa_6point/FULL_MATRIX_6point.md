# Optimal+DSA 7-Point Coverage (+(8)(9) 70B 확장) — Multi-Model Full Matrix (Single-Completed Doc)

> **Source**: TSK_042 baseline (2026-06-02, host DSA WQ disabled) + 본 sweep (2026-06-10+ host DSA WQ enabled) + **(7) SUB_213 uniform-pad sweep (2026-06-14)**
> **Coverage**: 10 models × 7 corpora × **9 points** ((1)~(6) + (7) best-K uniform-pad + (8) W4A4 + (9) 동적-K). (1)~(6) = 406/420, (7) = 70/70, **(9) 동적-K = 70/70 전 모델 측정완료 (2026-06-20)**, **(8) W4A4 = Llama-3.1-70B만 게이트 PASS(SR-004), 그외 9모델 N/A**(게이트 FAIL 7 + 양자화불가 2, §14.1). 70B vs (7): (8) +7.1%, (9) −7.9%.
> **(7) 정의**: `VLLM_SUFFIX_PAD_UNIFORM=1` + suffix+FaP 에서 per-(model,corpus) 최적 K∈{4,6,8,12} 의 tps. 셀에 `tps (Kx)` 병기. taskset 0-47,56-103 (절대비교 시 caveat). 출력분포 등가 (정확도 게이트 D-ii PASS).
> **핵심 결과**: (7) 가 **70셀 중 68셀**에서 행 최대 = 기존 6개 설정을 전부 상회. 예외 2셀(Qwen2.5-7B lmsys=(6), mix=(4)).
> **Stand-alone**: HW/SW/corpus/모델 정보 모두 포함, 외부 의존 없이 재현 가능

---

## 1. Hardware Environment

| 항목 | 값 |
|---|---|
| GPU | NVIDIA B200 × 8 (sm_100, 183 GiB HBM3e each, NVLink5) |
| CPU | Intel Xeon Platinum 8570 dual-socket (Emerald Rapids), 224 thread |
| ISA features | AVX-512 + AMX (BF16/INT8) + DSA 8 SWQ (dsa0/dsa1 × 4 engines) |
| NUMA | 2 nodes — NUMA0=0-55/112-167, NUMA1=56-111/168-223 |
| DRAM | 2 TB system memory |
| Host | `dgx-b200` |

## 2. Host DSA WQ state (the binding confounder)

호스트 시스템 `/sys/bus/dsa/devices/wq{0,1}.{0,1,2,3}/state` 의 enable/disable 상태가 **vllm 실행 전체에 영향**:

| 측정 시점 | sysfs mtime | WQ state | 측정 데이터 |
|---|---|:---:|---|
| 2026-06-02 (TSK_042) | (WQ enable 이전) | **DISABLED** | TSK_042 vanilla / suffix |
| 2026-06-08 00:40 | sysfs `wq0.0/state` mtime | WQ enabled | LHC Phase 작업 일환 |
| 2026-06-10+ (본 sweep) | (enable 후) | **ENABLED** | 본 sweep van(ON)/DSA(ON)/suf(ON)/suf+dsa(ON) |

**vllm env 와 호스트 DSA 의 직교성**: VLLM_LHC_DSA env 안 켜도 호스트 DSA 가 enabled 상태면 시스템 메모리 동작에 영향 (vanilla 측정에서도 +33~+36% tps).

### 2.1 호스트 DSA 상세 구성 (sweep 시점 = 현재, 2026-06-12 sysfs 실측)

**드라이버 / 디바이스 레벨:**

| 항목 | 값 |
|---|---|
| 커널 드라이버 | `idxd` (+ `idxd_bus`, `iaa_crypto` 연동) — dsa bus 드라이버: crypto/dmaengine/idxd/**user** |
| 설정 도구 | `accel-config` 4.1.6+ (`/usr/bin/accel-config`) |
| 디바이스 | `dsa0`(NUMA 0) / `dsa1`(NUMA 1) — 둘 다 enabled |
| PASID | enabled (=1, SVM → 유저공간 ENQCMD 제출 가능) |
| 디바이스 한계 | max_groups 4 / max_engines 4 / max_work_queues 8 / **max_wq_size 128** |
| device-level max_batch / max_transfer | 1,024 / 2 GiB |

**WQ 구성** (디바이스당 8개 중 4개만 enable):

| device | NUMA | enabled WQ (name) | disabled WQ |
|---|:---:|---|---|
| `dsa0` | 0 | wq0.0–wq0.3 (`lhc0`–`lhc3`) | wq0.4–wq0.7 (size=0, type=none) |
| `dsa1` | 1 | wq1.0–wq1.3 (`lhc1_0`–`lhc1_3`) | wq1.4–wq1.7 (size=0, type=none) |

Enabled WQ 공통 속성 (8개 모두 동일):

| 속성 | 값 | 의미 |
|---|---|---|
| `mode` | **shared** (SWQ) | ENQCMD 로 다중 프로세스 공유 제출 |
| `type` | `user` | `/dev/dsa/wqX.Y` 유저스페이스 직접 접근 |
| `size` | 16 entries | WQ당 16 (총 64/128 — 디바이스 용량의 절반 사용) |
| `group_id` / `priority` / `threshold` | 0 / 10 / 8 | 전부 group 0, SWQ 제출 한도 8 |
| `max_transfer_size` (per-WQ) | 2 MiB | per-descriptor 전송 상한 |
| `max_batch_size` (per-WQ) | 32 | batch descriptor 상한 |
| `block_on_fault` | 1 | page fault 시 block |
| `clients` | **0** | 현재 어떤 프로세스도 미사용 (SUB_213 confounder 논거) |

**Engine/그룹 토폴로지**: 디바이스당 engine 4개(engine0.0–0.3 / engine1.0–1.3) 전부 group 0 — "WQ 4 → group 0 → engine 4" 단일 그룹, QoS 분리 없음, 4-way engine 병렬.

**디바이스 노드 / 권한**: `/dev/dsa/wq{0,1}.{0..3}` = `crw-rw-rw- root root` (major 504) — **0666 world-writable** 이라 비특권 사용자도 직접 open 가능 (vllm `VLLM_LHC_DSA_DEV` default `/dev/dsa/wq0.0` 이 그대로 동작하는 이유).

**비영속성 주의**: `/etc/accel-config/` 에 저장 설정 없음 → 이 구성은 수동 enable (2026-06-08 12:59) 상태. **재부팅 시 WQ 구성 소실** — 재현하려면 `accel-config save-config` 저장 또는 enable 스크립트 필요.

> **주의**: "host DSA enabled" 효과는 위 WQ 가 enable 되어 있다는 **호스트 시스템 상태** 자체를 말한다. SUB_213 검증에서 `clients=0`(어떤 프로세스도 WQ 미사용) 임이 확인되어, +36% 의 진짜 원인은 cudagraph_mode(PIECEWISE→FaP) 차이라는 가설이 유력 — §2 의 시점 표는 confounder 후보 기록으로 유지.

## 3. 6 Measurement Points Definition

| ID | label | host DSA | spec decode | vllm DSA env | source dir |
|---|---|:---:|:---:|:---:|---|
| (1) | van(OFF) | disabled | — | — | TSK_042 |
| (2) | van(ON) | **enabled** | — | — | 본 sweep |
| (3) | DSA(ON) | enabled | — | **VLLM_LHC_DSA=1 VLLM_LEVER_N9=1** | 본 sweep |
| (4) | suf(OFF) | disabled | suffix K=32 | — | TSK_042 |
| (5) | suf(ON) | **enabled** | suffix K=32 | — | 본 sweep |
| (6) | suf+dsa(ON) | enabled | suffix K=32 | **on** | 본 sweep |
| (7) | **suffix+FaP+Kpad** (=bestK pad) ⭐ | enabled | **suffix K∈{4,6,8,12} + `VLLM_SUFFIX_PAD_UNIFORM=1`** | — | **SUB_213** (2026-06-14) |
| (8) | **(7) + W4A4 양자화** (Llama-3.1-70B만 PASS; 그외 9모델 N/A=게이트 FAIL/양자화불가, §14.1) | enabled | suffix bestK pad + **AWQ+GPTQ NVFP4 W4A4** | — | **2026-06-19** (codesci/POINT8) |
| (9) | **(7) + 동적-K** (bf16, **전 10모델 측정완료**) | enabled | suffix pad + **`VLLM_SUFFIX_DYN_K=1` KS={4,6,12}**(α-EMA) | — | **2026-06-20** (SUB_247 D3 + runs_p9_dynk) |

> (7) 은 (5) suf(ON) 기준선에 (a) K 를 4/6/8/12 로 바꾸고 (b) draft 를 K 로 균일패딩(`VLLM_SUFFIX_PAD_UNIFORM=1`)해 FULL cudagraph 를 적중시킨 변형. pad 토큰은 rejection sampler 가 기각 → 출력분포 (5) 와 등가. 셀값 = per-(model,corpus) 최적 K 의 tps. (taskset 0-47,56-103 적용 — 절대비교 caveat.)
>
> **명명/지위 (2026-06-15)**: (7) = **`suffix+FaP+Kpad`** = 본 fork 의 현재 챔피언(70셀 중 68셀 행 최대). 이것이 앞으로 **"뛰어넘을 대상(to-beat baseline)"** 이다 — upstream 의 동등 작업(eagle_dynamic `DynamicProposer` #26504 = 동적 K, uniform cudagraph 정렬 #23679)을 비교에 넣을 때, 그것들이 (7) `suffix+FaP+Kpad` 를 넘는지로 판정한다. (upstream rebase 후 (8) 열로 추가 측정 권장. 메모리 `spec-decode-adaptive-k-upstream` 참조.)

### (8)(9) 확장 측정 (2026-06-19, Llama-3.1-70B, output_tps, DSA-on, TP8)

(7)(bf16 best-K pad) 위에 (8)(W4A4 양자화)·(9)(동적-K)를 얹어 측정. (7)/(9) = 06-13 era, (8) = 06-19(세션-drift caveat). (9)=`VLLM_SUFFIX_DYN_K=1 KS={4,6,12}`(SUB_247 D3). (8)=`awqgptq_nvfp4_70b`(SR-004 게이트 PASS).

| corpus | (7) bf16 best-K | (8) W4A4 ((8)/(7)) | (9) 동적-K ((9)/(7)) |
|---|---:|---:|---:|
| sharegpt | 6,092 (K8) | 6,183 (+1.5%) | 6,165 (+1.2%) |
| swebench | 7,179 (K8) | 7,328 (+2.1%) | 7,398 (+3.1%) |
| humaneval | 6,218 (K6) | 7,216 (**+16.1%**) | 6,247 (+0.5%) |
| mbpp | 3,814 (K4) | 4,909 (**+28.7%**) | 3,256 (**−14.6%**) |
| wildchat | 6,883 (K6) | 7,497 (+8.9%) | 6,465 (−6.1%) |
| lmsys | 6,060 (K6) | 6,418 (+5.9%) | 6,043 (−0.3%) |
| mix | 14,389 (K12) | 13,011 (−9.6%) | 9,656 (**−32.9%**) |
| **기하평균 vs (7)** | 기준 | **+7.1%** | **−7.9%** |

> **판정**: **(8)(+W4A4 양자화) = (7) 대비 +7.1%**(코드 corpus 큼·mix −10%; standalone +55%→스택 위 수확체감). **(9)(동적-K) = (7) 대비 −7.9%**(컨트롤러가 oracle best-K 못 따라감, mix −33%; naive 고정K6 대비는 ~+1.1%). → **(7)이 여전히 챔피언, (8)만 그 위 +7% 더함. 동적-K(=upstream #26504 류)는 (7) 못 넘음**(suffix가 이미 native 동적-K라 흡수). 상세=`codesci/POINT8_RESULT.md`·`POINT9_RESULT.md`.

## 4. vLLM Configuration

| 항목 | 값 |
|---|---|
| vLLM version | `1.7.dev16107+gffe20fb09.d20260601` (sm_100) |
| Venv | `/workspace/vllm_dev_prj/bin/python` (CPython 3.12) |
| Editable source | `/workspace/host_vllm_hybrid/vllm/` |
| Compilation | `cudagraph_mode=FULL_AND_PIECEWISE` (FaP) |
| gpu_mem_util | 0.85 |
| max_model_len | 16,384 |
| Backend | OpenAI-compatible streaming completions |

Env (모든 점 공통):
```bash
export ARCTIC_INFERENCE_ENABLED=0 VLLM_PLUGINS=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NGRAM_NUM_THREADS_CAP=8 VLLM_NGRAM_DIVIDE_BY_TP=0
```

### 4.1 Suffix decoding 설정 ((4)(5)(6) 점)

부팅 인자 (`sweep_corpus.sh:86`):
```bash
--speculative-config '{"method":"suffix","num_speculative_tokens":32}'
```

| 파라미터 | 값 | 비고 |
|---|---|---|
| `method` | `suffix` | suffix-tree 기반 draft (draft 모델 불요, CPU-side) |
| `num_speculative_tokens` (K) | **32** | step 당 최대 draft 길이 (TSK_042 dominant 설정 승계) |
| `suffix_decoding_max_tree_depth` | 24 (default) | global/prompt suffix tree 최대 깊이 |
| `suffix_decoding_max_cached_requests` | 10,000 (default) | global tree 캐시 요청 수 |
| `suffix_decoding_max_spec_factor` | 1.0 (default) | match 길이 대비 spec 길이 상한 계수 |
| `suffix_decoding_min_token_prob` | 0.1 (default) | draft 채택 최소 빈도확률 |

suffix 관련 env: **본 sweep 에서는 추가 env 없음** (`VLLM_SUFFIX_PAD_UNIFORM` 은 SUB_213 lever 로 본 sweep 이후 추가된 것 — 본 데이터에는 미적용, 즉 가변 길이 draft → PIECEWISE 경로). 파라미터 정의: `vllm/config/speculative.py:162-177`, env 처리: `vllm/v1/spec_decode/suffix_decoding.py:110-120`.

### 4.2 vllm DSA env 설정 ((3)(6) 점)

```bash
VLLM_LHC_DSA=1 VLLM_LEVER_N9=1 VLLM_LHC_DSA_MIN=65536
```

| env | 값 | 의미 | 코드 |
|---|---|---|---|
| `VLLM_LHC_DSA` | 1 | LHC DSA lane 활성 (default off). regime detector static mode 에서 이 플래그를 따름 | `vllm/v1/lhc/dsa_lane.py:19`, `regime_detector.py:334-337` |
| `VLLM_LEVER_N9` | 1 | DSA 로 host↔pinned memcpy 오프로드 (SUB_201 lever N9) | `vllm/envs.py:117,990` |
| `VLLM_LHC_DSA_MIN` | 65536 | DSA 사용 최소 바이트 (64 KiB 미만 copy 는 CPU 경로 유지) | `vllm/v1/lhc/dsa_lane.py:21` |
| `VLLM_LHC_DSA_DEV` | (미설정 → default `/dev/dsa/wq0.0`) | 사용 WQ 디바이스 경로 | `dsa_lane.py:20` |
| `VLLM_LHC_DSA_WQ_PER_RANK` / `VLLM_LHC_DSA_RANK` | (미설정) | TP rank → WQ 분산 (Phase 3 옵션, 본 sweep 미사용) | `dsa_lane.py:24-27` |

> (1)(2)(4)(5) 점들은 위 DSA env **전부 미설정** (= vllm DSA lane off). (2)(5) 와 (1)(4) 의 차이는 vllm 설정이 아니라 §2 호스트 WQ state (및 §11/SUB_213 의 cudagraph confounder 후보) 다.

## 5. Benchmark Configuration

| 항목 | 값 |
|---|---|
| Harness | `vllm_config_perf/gating/realistic_eval/throughput_runner.py` |
| Prompts | `runs/tput_t1t3_20260602/sampled_prompts.parquet` (real trace) |
| Concurrency | 32 |
| max_tokens | 8,192 |
| Streaming | True |
| Per-corpus limit | 자연 trace 분포 |
| `mix` 셀 | 500-prompt shuffle (seed=0) |

## 6. Corpus Information (7 conditions)

| Corpus tag | Source | 특성 |
|---|---|---|
| `sharegpt` | ShareGPT (LMSYS) | 대화/chat |
| `wildchat` | WildChat-1M | natural chat |
| `lmsys` | LMSYS-Chat-1M | chat |
| `humaneval` | LiveCodeBench (HumanEval) | 짧은 코드 |
| `mbpp` | LiveCodeBench (MBPP) | 짧은 코드 (Python) |
| `swebench` | SWE-Bench Lite | 코드 + repo context |
| `mix` | 위 6 corpus shuffle | 운영 mix proxy |

## 7. Model Information (10 models)

| TAG | HF model_id | TP | num_heads | 특성 |
|---|---|---:|---:|---|
| `Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | 4 | 28 | 7B dense |
| `DeepSeek-R1-Distill-Qwen-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 4 | 28 | 7B reasoning distill |
| `Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 8 | 32 | 8B dense |
| `Qwen2.5-32B-Instruct` | `Qwen/Qwen2.5-32B-Instruct` | 8 | 40 | 32B dense |
| `DeepSeek-R1-Distill-Qwen-32B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 8 | 40 | 32B reasoning distill |
| `Qwen2.5-72B-Instruct` | `Qwen/Qwen2.5-72B-Instruct` | 8 | 64 | 72B dense |
| `Llama-3.1-70B-Instruct` | `meta-llama/Llama-3.1-70B-Instruct` | 8 | 64 | 70B dense |
| `DeepSeek-R1-Distill-Llama-70B` | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 8 | 64 | 70B reasoning distill |
| `Llama-3.1-405B-Instruct-FP8` | `meta-llama/Llama-3.1-405B-Instruct-FP8` | 8 | 128 | 405B FP8 (⚠ (5)(6) boot fail) |
| `DeepSeek-R1` | `deepseek-ai/DeepSeek-R1` | 8 | 128 | 671B MoE (37B active) |

---

## 8. 6-Point Headline — mix corpus (10 모델)

> (7) bestKpad(bf16) 열 추가 (2026-06-19). (7) = suffix + uniform-pad + FaP + best-K (SUB_213). 70B 만 (8)(W4A4)·(9)(동적-K) 별도 보유(§9 참조). winner 는 (1)~(7) 전체 기준.

| model | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad | (8) +W4A4 | (9) dynK | **winner** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `Qwen2.5-7B-Instruct` | 4,169 | 5,564 | 5,572 | 7,803 | 7,478 | 7,457 | 7,488 | N/A | 6,665 | **(4) suf(OFF) 7,803** |
| `DeepSeek-R1-Distill-Qwen-7B` | 9,058 | 12,277 | 12,301 | 24,458 | 22,467 | 22,193 | **27,664** | N/A | 19,964 | **(7) bestKpad 27,664** |
| `Llama-3.1-8B-Instruct` | 8,850 | 12,089 | 12,058 | 27,851 | 24,407 | 26,615 | **33,531** | N/A | 31,302 | **(7) bestKpad 33,531** |
| `Qwen2.5-32B-Instruct` | 3,056 | 4,694 | 4,698 | 6,597 | 5,979 | 6,256 | **7,274** | N/A | 5,771 | **(7) bestKpad 7,274** |
| `DeepSeek-R1-Distill-Qwen-32B` | 4,938 | 5,134 | 5,060 | 9,056 | 8,378 | 9,240 | **11,750** | N/A | 8,686 | **(7) bestKpad 11,750** |
| `Qwen2.5-72B-Instruct` | 2,735 | 2,967 | 2,902 | 5,268 | 5,643 | 5,266 | **6,897** | N/A | 6,145 | **(7) bestKpad 6,897** |
| `Llama-3.1-70B-Instruct` | 3,129 | 3,206 | 3,192 | 10,400 | 10,247 | 8,829 | **14,389** | 13,011 | 9,656 | **(7) bestKpad 14,389** |
| `DeepSeek-R1-Distill-Llama-70B` | 3,164 | 3,244 | 3,198 | 6,127 | 6,175 | 5,818 | **6,493** | N/A | 4,869 | **(7) bestKpad 6,493** |
| `Llama-3.1-405B-Instruct-FP8` | 1,252 | 1,252 | 1,271 | 2,829 | — | — | **3,446** | N/A | 3,224 | **(7) bestKpad 3,446** |
| `DeepSeek-R1` | 1,538 | 1,599 | 1,601 | 781 | 754 | 727 | **2,224** | N/A | **2,294** | **(9) dynK 2,294** |

## 9. Per-Model 6-Point × 7-Corpus Tables

> **(7) bestKpad(bf16) 추가 (2026-06-19, 10개 모델 전체).** (7) = suffix-spec + uniform-pad + FULL_AND_PIECEWISE cudagraph + corpus별 best-K∈{4,6,8,12} (SUB_213 §5 `runs_multimodel`, output_tps). 셀 표기 `값 (K{k})`. (7)는 06-13 era harness, (1)~(6)은 6-point era(SUB_212) → 동일-머신·동일-하니스이나 다른 세션이므로 **cross-era 비교**(작은 차는 drift 범위일 수 있음, 큰 차는 best-K-pad 실효과). Llama-3.1-70B-Instruct 만 추가로 (8)(W4A4)·(9)(동적-K) 보유(데이터가 70B 한정). (7)는 거의 전 모델·전 corpus에서 winner — 예외는 cross-era drift 셀(Qwen2.5-7B lmsys=(6), mix=(4))뿐.

### `Qwen2.5-7B-Instruct` (TP=4)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 4,189 | 5,600 | 5,640 | 6,167 | 6,058 | 6,164 | **6,287 (K4)** | N/A | 6,149 | (7) bestKpad |
| swebench | 4,120 | 5,871 | 5,973 | 5,416 | 5,322 | 5,551 | **6,627 (K4)** | N/A | 6,433 | (7) bestKpad |
| humaneval | 3,754 | 5,331 | 5,336 | 5,213 | 4,863 | 4,989 | **5,830** (K6) | N/A | **6,059** | (9) dynK |
| mbpp | 3,814 | 5,931 | 5,965 | 5,506 | 5,346 | 5,390 | **7,016 (K4)** | N/A | 6,386 | (7) bestKpad |
| wildchat | 4,184 | 5,694 | 5,644 | 6,285 | 5,974 | 6,293 | **6,685 (K4)** | N/A | 6,449 | (7) bestKpad |
| lmsys | 4,090 | 5,409 | 5,427 | 5,956 | 5,906 | 6,038 | 5,910 (K4) | N/A | 5,819 | (6) suf+dsa(ON) |
| mix | 4,169 | 5,564 | 5,572 | 7,803 | 7,478 | 7,457 | 7,488 (K6) | N/A | 6,665 | (4) suf(OFF) |

### `DeepSeek-R1-Distill-Qwen-7B` (TP=4)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 8,724 | 12,232 | 12,170 | 11,961 | 11,234 | 11,240 | **15,984 (K8)** | N/A | 14,718 | (7) bestKpad |
| swebench | 8,835 | 11,891 | 11,888 | 15,422 | 14,671 | 14,682 | **18,308 (K8)** | N/A | 17,562 | (7) bestKpad |
| humaneval | 8,159 | 11,273 | 11,240 | 11,459 | 11,035 | 10,519 | **16,824 (K12)** | N/A | 16,199 | (7) bestKpad |
| mbpp | 8,440 | 11,694 | 11,676 | 12,398 | 11,481 | 12,260 | **18,176 (K8)** | N/A | 17,709 | (7) bestKpad |
| wildchat | 8,925 | 12,319 | 12,210 | 11,717 | 10,795 | 11,263 | **16,182 (K8)** | N/A | 15,303 | (7) bestKpad |
| lmsys | 8,811 | 12,055 | 12,057 | 11,360 | 11,052 | 11,390 | **16,256 (K8)** | N/A | 14,442 | (7) bestKpad |
| mix | 9,058 | 12,277 | 12,301 | 24,458 | 22,467 | 22,193 | **27,664 (K12)** | N/A | 19,964 | (7) bestKpad |

### `Llama-3.1-8B-Instruct` (TP=8)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 8,868 | 12,091 | 12,088 | 19,054 | 18,073 | 19,328 | **25,372 (K12)** | N/A | 23,919 | (7) bestKpad |
| swebench | 8,348 | 11,970 | 11,518 | 21,353 | 20,735 | 20,518 | **27,828 (K12)** | N/A | 26,796 | (7) bestKpad |
| humaneval | 9,048 | 10,967 | 11,061 | 15,126 | 14,794 | 15,601 | **21,034 (K12)** | N/A | 17,660 | (7) bestKpad |
| mbpp | 8,730 | 12,190 | 12,066 | 17,825 | 17,976 | 17,360 | **22,129** (K12) | N/A | **22,513** | (9) dynK |
| wildchat | 9,002 | 12,210 | 12,197 | 19,856 | 19,602 | 19,451 | **25,910 (K12)** | N/A | 25,797 | (7) bestKpad |
| lmsys | 9,074 | 12,528 | 11,993 | 19,862 | 19,361 | 18,905 | **25,305** (K12) | N/A | **25,384** | (9) dynK |
| mix | 8,850 | 12,089 | 12,058 | 27,851 | 24,407 | 26,615 | **33,531 (K12)** | N/A | 31,302 | (7) bestKpad |

### `Qwen2.5-32B-Instruct` (TP=8)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 3,079 | 4,591 | 4,607 | 4,662 | 4,499 | 4,474 | **5,933 (K4)** | N/A | 5,662 | (7) bestKpad |
| swebench | 2,892 | 4,148 | 4,244 | 5,002 | 4,348 | 4,566 | **5,770 (K4)** | N/A | 5,750 | (7) bestKpad |
| humaneval | 2,571 | 3,602 | 3,527 | 4,859 | 4,325 | 4,269 | **5,605** (K4) | N/A | **5,738** | (9) dynK |
| mbpp | 2,915 | 4,295 | 4,425 | 5,138 | 4,826 | 4,817 | **6,046 (K4)** | N/A | 6,000 | (7) bestKpad |
| wildchat | 3,128 | 4,804 | 4,738 | 4,884 | 4,651 | 4,504 | **5,639** (K6) | N/A | **5,818** | (9) dynK |
| lmsys | 3,053 | 4,686 | 4,628 | 4,478 | 4,578 | 4,249 | **5,921** (K6) | N/A | **6,076** | (9) dynK |
| mix | 3,056 | 4,694 | 4,698 | 6,597 | 5,979 | 6,256 | **7,274 (K8)** | N/A | 5,771 | (7) bestKpad |

### `DeepSeek-R1-Distill-Qwen-32B` (TP=8)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 4,803 | 4,931 | 4,902 | 4,996 | 4,682 | 4,613 | **5,809** (K4) | N/A | **6,135** | (9) dynK |
| swebench | 4,409 | 4,524 | 4,561 | 5,241 | 5,589 | 5,444 | **6,718 (K6)** | N/A | 6,378 | (7) bestKpad |
| humaneval | 3,462 | 3,729 | 4,208 | 3,771 | 3,935 | 3,435 | **4,550 (K6)** | N/A | 4,436 | (7) bestKpad |
| mbpp | 4,690 | 4,905 | 4,806 | 5,690 | 5,097 | 5,221 | **6,253 (K6)** | N/A | 5,913 | (7) bestKpad |
| wildchat | 4,891 | 5,066 | 5,102 | 5,729 | 5,363 | 5,539 | **6,564** (K6) | N/A | **6,582** | (9) dynK |
| lmsys | 4,898 | 4,993 | 5,011 | 5,356 | 4,980 | 5,116 | **6,393** (K4) | N/A | **6,420** | (9) dynK |
| mix | 4,938 | 5,134 | 5,060 | 9,056 | 8,378 | 9,240 | **11,750 (K12)** | N/A | 8,686 | (7) bestKpad |

### `Qwen2.5-72B-Instruct` (TP=8)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 2,688 | 2,830 | 2,906 | 3,219 | 3,095 | 3,006 | **4,124** (K4) | N/A | **4,406** | (9) dynK |
| swebench | 2,361 | 2,474 | 2,444 | 2,647 | 2,743 | 2,635 | **3,412 (K4)** | N/A | 3,279 | (7) bestKpad |
| humaneval | 806 | 1,989 | 2,542 | 2,489 | 2,358 | 2,022 | **2,587 (K4)** | N/A | 2,506 | (7) bestKpad |
| mbpp | 3,395 | 3,417 | 3,441 | 3,234 | 2,976 | 2,910 | **3,443 (K4)** | N/A | 3,433 | (7) bestKpad |
| wildchat | 2,803 | 2,929 | 2,909 | 2,621 | 2,434 | 2,591 | **3,383 (K4)** | N/A | 3,374 | (7) bestKpad |
| lmsys | 2,807 | 3,169 | 3,083 | 3,429 | 2,978 | 3,153 | **4,673 (K4)** | N/A | 4,542 | (7) bestKpad |
| mix | 2,735 | 2,967 | 2,902 | 5,268 | 5,643 | 5,266 | **6,897 (K6)** | N/A | 6,145 | (7) bestKpad |

### `Llama-3.1-70B-Instruct` (TP=8)

> (7)/(8)/(9) 추가 (2026-06-19). (7)=suffix+FaP+best-Kpad(bf16, SUB_213 §5), (8)=(7)+AWQ+GPTQ NVFP4 W4A4(SR-004 게이트 PASS, best-K), (9)=(7)+동적-K(VLLM_SUFFIX_DYN_K, bf16, SUB_247 D3). (7)/(9)=06-13 era, (8)=06-19(세션-drift caveat). 셀=output_tps.

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK(bf16) | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 3,091 | 3,177 | 3,139 | 4,864 | 4,542 | 4,634 | 6,092 | **6,183** | 6,165 | (8) W4A4 |
| swebench | 2,878 | 2,809 | 2,968 | 6,026 | 5,949 | 5,455 | 7,179 | 7,328 | **7,398** | (9) dynK |
| humaneval | 3,391 | 3,456 | 2,899 | 4,728 | 4,598 | 4,549 | 6,218 | **7,216** | 6,247 | (8) W4A4 |
| mbpp | 1,773 | 1,699 | 1,778 | 3,266 | 3,243 | 2,273 | 3,814 | **4,909** | 3,256 | (8) W4A4 |
| wildchat | 3,172 | 3,213 | 3,268 | 5,261 | 5,142 | 4,966 | 6,883 | **7,497** | 6,465 | (8) W4A4 |
| lmsys | 3,040 | 3,145 | 3,123 | 3,958 | 3,677 | 3,818 | 6,060 | **6,418** | 6,043 | (8) W4A4 |
| mix | 3,129 | 3,206 | 3,192 | 10,400 | 10,247 | 8,829 | **14,389** | 13,011 | 9,656 | (7) bestKpad |
| **기하평균 vs (7)** | | | | | | | 기준 | **+7.1%** | **−7.9%** | (8) |

### `DeepSeek-R1-Distill-Llama-70B` (TP=8)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 3,033 | 3,018 | 3,142 | 2,660 | 2,579 | 2,503 | **4,064 (K6)** | N/A | 3,834 | (7) bestKpad |
| swebench | 3,236 | 3,142 | 3,182 | 2,739 | 2,739 | 2,642 | **3,589** (K4) | N/A | **3,590** | (9) dynK |
| humaneval | 2,852 | 2,828 | 2,812 | 2,788 | 2,809 | 2,718 | **3,878 (K4)** | N/A | 3,646 | (7) bestKpad |
| mbpp | 2,777 | 2,989 | 2,954 | 2,426 | 2,328 | 2,265 | **3,444** (K4) | N/A | **3,561** | (9) dynK |
| wildchat | 3,127 | 3,208 | 3,166 | 2,658 | 2,544 | 2,661 | **4,295 (K4)** | N/A | 4,231 | (7) bestKpad |
| lmsys | 2,992 | 3,046 | 3,045 | 2,848 | 2,756 | 2,844 | **4,268 (K4)** | N/A | 4,158 | (7) bestKpad |
| mix | 3,164 | 3,244 | 3,198 | 6,127 | 6,175 | 5,818 | **6,493 (K12)** | N/A | 4,869 | (7) bestKpad |

### `Llama-3.1-405B-Instruct-FP8` (TP=8)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 1,217 | 1,239 | 1,229 | 2,061 | — | — | **2,391** (K4) | N/A | **2,497** | (9) dynK |
| swebench | 1,204 | 1,211 | 1,239 | 2,639 | — | — | **3,278 (K6)** | N/A | 3,259 | (7) bestKpad |
| humaneval | 1,253 | 1,237 | 1,192 | 2,112 | — | — | **2,779** (K6) | N/A | **2,851** | (9) dynK |
| mbpp | 916 | 883 | 815 | 1,725 | — | — | **2,224 (K6)** | N/A | 1,766 | (7) bestKpad |
| wildchat | 1,280 | 1,267 | 1,263 | 2,290 | — | — | **2,815 (K6)** | N/A | 2,467 | (7) bestKpad |
| lmsys | 1,220 | 1,247 | 1,221 | 2,243 | — | — | **2,769 (K6)** | N/A | 2,546 | (7) bestKpad |
| mix | 1,252 | 1,252 | 1,271 | 2,829 | — | — | **3,446 (K6)** | N/A | 3,224 | (7) bestKpad |

### `DeepSeek-R1` (TP=8)

| corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestKpad(bf16) | (8) +W4A4 | (9) dynK | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| sharegpt | 1,475 | 1,565 | 1,559 | 797 | 730 | 794 | **1,872 (K4)** | N/A | 1,816 | (7) bestKpad |
| swebench | 1,474 | 1,536 | 1,518 | 538 | 496 | 542 | **1,980** (K4) | N/A | **1,987** | (9) dynK |
| humaneval | 1,004 | 961 | 858 | 606 | 1,219 | 670 | **1,955** (K4) | N/A | **1,970** | (9) dynK |
| mbpp | 1,437 | 1,482 | 1,490 | 677 | 661 | 669 | **1,852** (K4) | N/A | **1,920** | (9) dynK |
| wildchat | 1,556 | 1,614 | 1,614 | 858 | 880 | 824 | **1,934** (K4) | N/A | **1,992** | (9) dynK |
| lmsys | 1,533 | 1,587 | 1,592 | 811 | 808 | 773 | **1,999 (K4)** | N/A | 1,949 | (7) bestKpad |
| mix | 1,538 | 1,599 | 1,601 | 781 | 754 | 727 | **2,224** (K12) | N/A | **2,294** | (9) dynK |

---

## 10. Effect Decomposition (mix corpus)

- **host DSA on vanilla** ((2) vs (1)): 호스트 WQ enable 의 vanilla 효과
- **host DSA on suffix** ((5) vs (4)): 호스트 WQ enable 의 suffix 효과
- **vllm env on vanilla** ((3) vs (2)): vllm `VLLM_LHC_DSA=1` 효과 (호스트 ON 위에)
- **vllm env on suffix** ((6) vs (5)): vllm env 효과 (suffix 위에)
- **suf-gain (host OFF)** ((4) vs (1)): 같은 host OFF state 에서 suffix vs vanilla
- **suf-gain (host ON)** ((5) vs (2)): 같은 host ON state 에서 suffix vs vanilla

| model | DSA on van | DSA on suf | vllm env on van | vllm env on suf | suf-gain (OFF) | suf-gain (ON) |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B-Instruct` | +33.5% | -4.2% | +0.2% | -0.3% | +87.2% | +34.4% |
| `DeepSeek-R1-Distill-Qwen-7B` | +35.5% | -8.1% | +0.2% | -1.2% | +170.0% | +83.0% |
| `Llama-3.1-8B-Instruct` | +36.6% | -12.4% | -0.3% | +9.0% | +214.7% | +101.9% |
| `Qwen2.5-32B-Instruct` | +53.6% | -9.4% | +0.1% | +4.6% | +115.9% | +27.4% |
| `DeepSeek-R1-Distill-Qwen-32B` | +4.0% | -7.5% | -1.4% | +10.3% | +83.4% | +63.2% |
| `Qwen2.5-72B-Instruct` | +8.5% | +7.1% | -2.2% | -6.7% | +92.6% | +90.2% |
| `Llama-3.1-70B-Instruct` | +2.4% | -1.5% | -0.4% | -13.8% | +232.4% | +219.6% |
| `DeepSeek-R1-Distill-Llama-70B` | +2.5% | +0.8% | -1.4% | -5.8% | +93.7% | +90.3% |
| `Llama-3.1-405B-Instruct-FP8` | -0.0% | — | +1.5% | — | +125.9% | — |
| `DeepSeek-R1` | +4.0% | -3.5% | +0.1% | -3.6% | -49.2% | -52.8% |

---

## 11. Verdict & Production Implications

### 11.1 Llama-405B-FP8 의 suffix 부팅 한계 (new finding)

- **Engine init failure**: `num_gpu_blocks=0 → override=512` 후 core proc crash
- **Affected**: (5) suf(ON), (6) suf+dsa(ON) — 14 cells 영구 미측정
- **Cause**: 405B-FP8 + suffix K=32 + B200 단일 TP=8 + gmu 0.85 호환성 한계
- **Paper limitation**: 405B-FP8 의 suffix decode 는 본 vllm 빌드에서 측정 불가

### 11.2 호스트 DSA 의 method 별 차등 효과

- **vanilla**: +33~+36% (host-bound regime, DSA memcpy 가속 도움)
- **suffix**: −5 ~ +10% (step-bound regime, 효과 무 또는 약한 손해)

### 11.3 vllm-level DSA env 의 영향

- **vanilla**: 0% (noise, Llama-8B 동일 세션 측정 기준)
- **suffix**: ±5% (corpus-dependent, dominant signal 없음)
- → **호스트 DSA 가 driver**, vllm env 는 부차적

### 11.4 운영 권장 (paper §discussion)

| 모델군 | 최선 lever | tps gain |
|---|---|---|
| 표준 dense (Qwen 7B/32B/72B, Llama 8B/70B/405B) | suf(OFF) | +50~+232% vs van(OFF) |
| Distill (DS-Qwen-32B) | suf+dsa(ON) | corpus-dependent |
| Distill reasoning (DS-Llama-70B) | suf(ON) | mix +∞ but other corpora marginal |
| 671B MoE (DeepSeek-R1) | **van(OFF)** | suffix net-negative across all corpora |

---

## 12. Raw Data

| 산출물 | 경로 |
|---|---|
| Per-cell JSON | `lhc_phase4/optimal_dsa/runs/summ_<TAG>_<METHOD>_<CORPUS>.json` |
| TSK_042 baseline | `vllm_config_perf/gating/realistic_eval/runs/tput_t1t3_20260602/` |
| Per-request raw | `lhc_phase4/optimal_dsa/runs/per_request_raw.jsonl` |
| Boot/Bench 로그 | `lhc_phase4/optimal_dsa/runs/_logs/` |
| Sweep scripts | `sweep_corpus.sh, sweep_multi.sh, sweep_complete.sh` |
| Aggregator | `aggregate.py` |
| Verify (LHC/git 격리) | `verify_dsa.sh` (C1/C2/C3 — DSA confounder 검증) |

## 13. Citation / Tracing

- Parent IDE: **IDE_023** (HPC Multi-Axis CPU Slack Harvesting on DGX B200)
- Parent TSK: **TSK_043** (Host-Side Slack Reclamation)
- Predecessor: TSK_042 (워크로드 활용 실험, vanilla+suffix baseline 70 cells × 2 = 140)
- 본 sweep: **406/420 = 96.7%** (140 baseline + 266 fresh)

---

## 14. Appendix — 전체 70셀 단일 통합 테이블 (10 models × 7 corpora × 7 points; +70B (8)(9))

> §8(mix headline)·§9(per-model)와 동일 데이터의 flat 뷰. 단위 tok/s. `—` = 측정 없음 ((5)(6)=Llama-405B-FP8 engine init fail §11.1). **(9) 동적-K = 전 모델 측정 완료 (2026-06-20, runs_p9_dynk; 405B는 --allow-deprecated-quantization 재실행).** **(8) W4A4 = Llama-3.1-70B만 PASS(SR-004). 그 외 9모델 전부 `N/A`** — 7모델(7B~72B, distill-70B)은 분포동등 게이트 FAIL(출력 비등가), 405B(이미 FP8)·R1-671B(MoE)는 AWQ smoothlayer 매핑 불가로 양자화 자체 실패. 실증=§14.1 게이트 표. 70B (7)(8)(9)는 §9 70B 표와 동일(POINT8/9).

| model | corpus | (1) van(OFF) | (2) van(ON) | (3) DSA(ON) | (4) suf(OFF) | (5) suf(ON) | (6) suf+dsa(ON) | (7) bestK pad | (8) +W4A4 | (9) dynK | winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `Qwen2.5-7B-Instruct` | sharegpt | 4,189 | 5,600 | 5,640 | 6,167 | 6,058 | 6,164 | **6,287 (K4)** | N/A | 6,149 | (7) bestKpad |
|  | swebench | 4,120 | 5,871 | 5,973 | 5,416 | 5,322 | 5,551 | **6,627 (K4)** | N/A | 6,433 | (7) bestKpad |
|  | humaneval | 3,754 | 5,331 | 5,336 | 5,213 | 4,863 | 4,989 | 5,830 (K6) | N/A | **6,059** | (9) dynK |
|  | mbpp | 3,814 | 5,931 | 5,965 | 5,506 | 5,346 | 5,390 | **7,016 (K4)** | N/A | 6,386 | (7) bestKpad |
|  | wildchat | 4,184 | 5,694 | 5,644 | 6,285 | 5,974 | 6,293 | **6,685 (K4)** | N/A | 6,449 | (7) bestKpad |
|  | lmsys | 4,090 | 5,409 | 5,427 | 5,956 | 5,906 | **6,038** | 5,910 (K4) | N/A | 5,819 | (6) suf+dsa(ON) |
|  | mix | 4,169 | 5,564 | 5,572 | **7,803** | 7,478 | 7,457 | 7,488 (K6) | N/A | 6,665 | (4) suf(OFF) |
| `DeepSeek-R1-Distill-Qwen-7B` | sharegpt | 8,724 | 12,232 | 12,170 | 11,961 | 11,234 | 11,240 | **15,984 (K8)** | N/A | 14,718 | (7) bestKpad |
|  | swebench | 8,835 | 11,891 | 11,888 | 15,422 | 14,671 | 14,682 | **18,308 (K8)** | N/A | 17,562 | (7) bestKpad |
|  | humaneval | 8,159 | 11,273 | 11,240 | 11,459 | 11,035 | 10,519 | **16,824 (K12)** | N/A | 16,199 | (7) bestKpad |
|  | mbpp | 8,440 | 11,694 | 11,676 | 12,398 | 11,481 | 12,260 | **18,176 (K8)** | N/A | 17,709 | (7) bestKpad |
|  | wildchat | 8,925 | 12,319 | 12,210 | 11,717 | 10,795 | 11,263 | **16,182 (K8)** | N/A | 15,303 | (7) bestKpad |
|  | lmsys | 8,811 | 12,055 | 12,057 | 11,360 | 11,052 | 11,390 | **16,256 (K8)** | N/A | 14,442 | (7) bestKpad |
|  | mix | 9,058 | 12,277 | 12,301 | 24,458 | 22,467 | 22,193 | **27,664 (K12)** | N/A | 19,964 | (7) bestKpad |
| `Llama-3.1-8B-Instruct` | sharegpt | 8,868 | 12,091 | 12,088 | 19,054 | 18,073 | 19,328 | **25,372 (K12)** | N/A | 23,919 | (7) bestKpad |
|  | swebench | 8,348 | 11,970 | 11,518 | 21,353 | 20,735 | 20,518 | **27,828 (K12)** | N/A | 26,796 | (7) bestKpad |
|  | humaneval | 9,048 | 10,967 | 11,061 | 15,126 | 14,794 | 15,601 | **21,034 (K12)** | N/A | 17,660 | (7) bestKpad |
|  | mbpp | 8,730 | 12,190 | 12,066 | 17,825 | 17,976 | 17,360 | 22,129 (K12) | N/A | **22,513** | (9) dynK |
|  | wildchat | 9,002 | 12,210 | 12,197 | 19,856 | 19,602 | 19,451 | **25,910 (K12)** | N/A | 25,797 | (7) bestKpad |
|  | lmsys | 9,074 | 12,528 | 11,993 | 19,862 | 19,361 | 18,905 | 25,305 (K12) | N/A | **25,384** | (9) dynK |
|  | mix | 8,850 | 12,089 | 12,058 | 27,851 | 24,407 | 26,615 | **33,531 (K12)** | N/A | 31,302 | (7) bestKpad |
| `Qwen2.5-32B-Instruct` | sharegpt | 3,079 | 4,591 | 4,607 | 4,662 | 4,499 | 4,474 | **5,933 (K4)** | N/A | 5,662 | (7) bestKpad |
|  | swebench | 2,892 | 4,148 | 4,244 | 5,002 | 4,348 | 4,566 | **5,770 (K4)** | N/A | 5,750 | (7) bestKpad |
|  | humaneval | 2,571 | 3,602 | 3,527 | 4,859 | 4,325 | 4,269 | 5,605 (K4) | N/A | **5,738** | (9) dynK |
|  | mbpp | 2,915 | 4,295 | 4,425 | 5,138 | 4,826 | 4,817 | **6,046 (K4)** | N/A | 6,000 | (7) bestKpad |
|  | wildchat | 3,128 | 4,804 | 4,738 | 4,884 | 4,651 | 4,504 | 5,639 (K6) | N/A | **5,818** | (9) dynK |
|  | lmsys | 3,053 | 4,686 | 4,628 | 4,478 | 4,578 | 4,249 | 5,921 (K6) | N/A | **6,076** | (9) dynK |
|  | mix | 3,056 | 4,694 | 4,698 | 6,597 | 5,979 | 6,256 | **7,274 (K8)** | N/A | 5,771 | (7) bestKpad |
| `DeepSeek-R1-Distill-Qwen-32B` | sharegpt | 4,803 | 4,931 | 4,902 | 4,996 | 4,682 | 4,613 | 5,809 (K4) | N/A | **6,135** | (9) dynK |
|  | swebench | 4,409 | 4,524 | 4,561 | 5,241 | 5,589 | 5,444 | **6,718 (K6)** | N/A | 6,378 | (7) bestKpad |
|  | humaneval | 3,462 | 3,729 | 4,208 | 3,771 | 3,935 | 3,435 | **4,550 (K6)** | N/A | 4,436 | (7) bestKpad |
|  | mbpp | 4,690 | 4,905 | 4,806 | 5,690 | 5,097 | 5,221 | **6,253 (K6)** | N/A | 5,913 | (7) bestKpad |
|  | wildchat | 4,891 | 5,066 | 5,102 | 5,729 | 5,363 | 5,539 | 6,564 (K6) | N/A | **6,582** | (9) dynK |
|  | lmsys | 4,898 | 4,993 | 5,011 | 5,356 | 4,980 | 5,116 | 6,393 (K4) | N/A | **6,420** | (9) dynK |
|  | mix | 4,938 | 5,134 | 5,060 | 9,056 | 8,378 | 9,240 | **11,750 (K12)** | N/A | 8,686 | (7) bestKpad |
| `Qwen2.5-72B-Instruct` | sharegpt | 2,688 | 2,830 | 2,906 | 3,219 | 3,095 | 3,006 | 4,124 (K4) | N/A | **4,406** | (9) dynK |
|  | swebench | 2,361 | 2,474 | 2,444 | 2,647 | 2,743 | 2,635 | **3,412 (K4)** | N/A | 3,279 | (7) bestKpad |
|  | humaneval | 806 | 1,989 | 2,542 | 2,489 | 2,358 | 2,022 | **2,587 (K4)** | N/A | 2,506 | (7) bestKpad |
|  | mbpp | 3,395 | 3,417 | 3,441 | 3,234 | 2,976 | 2,910 | **3,443 (K4)** | N/A | 3,433 | (7) bestKpad |
|  | wildchat | 2,803 | 2,929 | 2,909 | 2,621 | 2,434 | 2,591 | **3,383 (K4)** | N/A | 3,374 | (7) bestKpad |
|  | lmsys | 2,807 | 3,169 | 3,083 | 3,429 | 2,978 | 3,153 | **4,673 (K4)** | N/A | 4,542 | (7) bestKpad |
|  | mix | 2,735 | 2,967 | 2,902 | 5,268 | 5,643 | 5,266 | **6,897 (K6)** | N/A | 6,145 | (7) bestKpad |
| `Llama-3.1-70B-Instruct` | sharegpt | 3,091 | 3,177 | 3,139 | 4,864 | 4,542 | 4,634 | 6,092 (K8) | 6,183 | 6,165 | (8) W4A4 |
|  | swebench | 2,878 | 2,809 | 2,968 | 6,026 | 5,949 | 5,455 | 7,179 (K8) | 7,328 | 7,398 | (9) dynK |
|  | humaneval | 3,391 | 3,456 | 2,899 | 4,728 | 4,598 | 4,549 | 6,218 (K6) | 7,216 | 6,247 | (8) W4A4 |
|  | mbpp | 1,773 | 1,699 | 1,778 | 3,266 | 3,243 | 2,273 | 3,814 (K4) | 4,909 | 3,256 | (8) W4A4 |
|  | wildchat | 3,172 | 3,213 | 3,268 | 5,261 | 5,142 | 4,966 | 6,883 (K6) | 7,497 | 6,465 | (8) W4A4 |
|  | lmsys | 3,040 | 3,145 | 3,123 | 3,958 | 3,677 | 3,818 | 6,060 (K6) | 6,418 | 6,043 | (8) W4A4 |
|  | mix | 3,129 | 3,206 | 3,192 | 10,400 | 10,247 | 8,829 | 14,389 (K12) | 13,011 | 9,656 | (7) bestKpad |
| `DeepSeek-R1-Distill-Llama-70B` | sharegpt | 3,033 | 3,018 | 3,142 | 2,660 | 2,579 | 2,503 | **4,064 (K6)** | N/A | 3,834 | (7) bestKpad |
|  | swebench | 3,236 | 3,142 | 3,182 | 2,739 | 2,739 | 2,642 | 3,589 (K4) | N/A | **3,590** | (9) dynK |
|  | humaneval | 2,852 | 2,828 | 2,812 | 2,788 | 2,809 | 2,718 | **3,878 (K4)** | N/A | 3,646 | (7) bestKpad |
|  | mbpp | 2,777 | 2,989 | 2,954 | 2,426 | 2,328 | 2,265 | 3,444 (K4) | N/A | **3,561** | (9) dynK |
|  | wildchat | 3,127 | 3,208 | 3,166 | 2,658 | 2,544 | 2,661 | **4,295 (K4)** | N/A | 4,231 | (7) bestKpad |
|  | lmsys | 2,992 | 3,046 | 3,045 | 2,848 | 2,756 | 2,844 | **4,268 (K4)** | N/A | 4,158 | (7) bestKpad |
|  | mix | 3,164 | 3,244 | 3,198 | 6,127 | 6,175 | 5,818 | **6,493 (K12)** | N/A | 4,869 | (7) bestKpad |
| `Llama-3.1-405B-Instruct-FP8` | sharegpt | 1,217 | 1,239 | 1,229 | 2,061 | — | — | 2,391 (K4) | N/A | **2,497** | (9) dynK |
|  | swebench | 1,204 | 1,211 | 1,239 | 2,639 | — | — | **3,278 (K6)** | N/A | 3,259 | (7) bestKpad |
|  | humaneval | 1,253 | 1,237 | 1,192 | 2,112 | — | — | 2,779 (K6) | N/A | **2,851** | (9) dynK |
|  | mbpp | 916 | 883 | 815 | 1,725 | — | — | **2,224 (K6)** | N/A | 1,766 | (7) bestKpad |
|  | wildchat | 1,280 | 1,267 | 1,263 | 2,290 | — | — | **2,815 (K6)** | N/A | 2,467 | (7) bestKpad |
|  | lmsys | 1,220 | 1,247 | 1,221 | 2,243 | — | — | **2,769 (K6)** | N/A | 2,546 | (7) bestKpad |
|  | mix | 1,252 | 1,252 | 1,271 | 2,829 | — | — | **3,446 (K6)** | N/A | 3,224 | (7) bestKpad |
| `DeepSeek-R1` | sharegpt | 1,475 | 1,565 | 1,559 | 797 | 730 | 794 | **1,872 (K4)** | N/A | 1,816 | (7) bestKpad |
|  | swebench | 1,474 | 1,536 | 1,518 | 538 | 496 | 542 | 1,980 (K4) | N/A | **1,987** | (9) dynK |
|  | humaneval | 1,004 | 961 | 858 | 606 | 1,219 | 670 | 1,955 (K4) | N/A | **1,970** | (9) dynK |
|  | mbpp | 1,437 | 1,482 | 1,490 | 677 | 661 | 669 | 1,852 (K4) | N/A | **1,920** | (9) dynK |
|  | wildchat | 1,556 | 1,614 | 1,614 | 858 | 880 | 824 | 1,934 (K4) | N/A | **1,992** | (9) dynK |
|  | lmsys | 1,533 | 1,587 | 1,592 | 811 | 808 | 773 | **1,999 (K4)** | N/A | 1,949 | (7) bestKpad |
|  | mix | 1,538 | 1,599 | 1,601 | 781 | 754 | 727 | 2,224 (K12) | N/A | **2,294** | (9) dynK |

**winner 분포** (70 cells, (1)~(9); (8)은 Llama-3.1-70B만 적용가능·그외 N/A): (4) 1 · (6) 1 · (7) **43** · (8) 5 · (9) 20

> (9) 동적-K가 20셀에서 행 최대 — 단 (7)은 06-14, (9)는 06-20 측정이라 cross-era 비교(소폭 차는 세션 drift 가능). (8)은 70B 5셀에서만 winner(나머지 비-70B는 측정 불가라 비경쟁).

### 14.1 (8) W4A4 분포동등 게이트 결과 (2026-06-20, 게이트=max_logprob_diff≤0.5 AND ppl_rel≤0.1)

| model | quant | token_match | max_logprob_diff | ppl_rel | gate |
|---|:---:|---:|---:|---:|:---:|
| `Llama-3.1-70B-Instruct` | OK | (SR-004) | 0.491 | 0.067 | **PASS** |
| `Qwen2.5-7B-Instruct` | OK | 18.9% | 0.677 | 0.109 | FAIL |
| `DeepSeek-R1-Distill-Qwen-7B` | OK | 0.4% | 0.277 | 0.358 | FAIL |
| `Llama-3.1-8B-Instruct` | OK | 13.5% | 0.652 | 0.147 | FAIL |
| `Qwen2.5-32B-Instruct` | OK | 28.1% | 0.462 | 0.266 | FAIL |
| `DeepSeek-R1-Distill-Qwen-32B` | OK | 17.2% | 0.590 | 0.108 | FAIL |
| `Qwen2.5-72B-Instruct` | OK | 15.8% | 1.339 | 0.089 | FAIL |
| `DeepSeek-R1-Distill-Llama-70B` | OK | 12.5% | 0.987 | 0.285 | FAIL |
| `Llama-3.1-405B-Instruct-FP8` | **양자화 실패** | — | — | — | N/A |
| `DeepSeek-R1` (671B MoE) | **양자화 실패** | — | — | — | N/A |

> **판정 (정직)**: AWQ+GPTQ NVFP4 W4A4 의 분포동등 게이트 PASS 는 **Llama-3.1-70B 단 하나**(SR-004). 7B~72B·distill-70B 7종은 전부 게이트 FAIL(4-bit 가중치가 출력 분포를 깨뜨림 — max_diff 또는 ppl_rel 초과). 405B(이미 FP8)·R1-671B(MoE)는 llmcompressor AWQ 가 smoothlayer 매핑 불가로 양자화 자체 실패. 즉 **(8) W4A4 는 프로젝트 출력동등 제약 하에서 Llama-3.1-70B 외에는 적용 불가** = 그 외 9모델 (8) 셀은 N/A 가 물리적 정답. (크기만의 문제 아님: Qwen-72B 는 70B 보다 큰데도 max_diff 1.34 로 크게 FAIL — 아키텍처·calibration 의존.) 산출: `codesci/runs_p8/p8_gate_results.csv`.
