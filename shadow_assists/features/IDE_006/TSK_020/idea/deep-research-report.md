# mystous/vllm_hybrid feat/spec-decode-tuning의 TSK_020 베스트 구성과 코딩 워크로드용 CPU 병렬성 분석

## 핵심 요약

이 보고서는 먼저 `api_tool`의 GitHub 커넥터 **GitHub**를 사용해 `mystous/vllm_hybrid` 저장소의 `feat/spec-decode-tuning` 브랜치를 조사했고, 추가 웹 검색은 하지 않았다. 현재 저장소가 문서·측정·result artifact로 고정해 둔 TSK_020의 현 베스트는 `SUB_047 t3`이며, 핵심 값은 `speculative_config={"method":"ngram","num_speculative_tokens":7,"prompt_lookup_max":5,"prompt_lookup_min":2}`, `tensor_parallel_size=8`, `max_model_len=16384`, `max_num_seqs=256`, `gpu_memory_utilization=0.85`, `kv_cache_dtype="fp8"`, `max_num_batched_tokens=8192`, `seed=0`, 그리고 환경변수 `VLLM_NGRAM_NUM_THREADS_CAP=8`, `VLLM_NGRAM_DIVIDE_BY_TP=0`이다. 이 조합은 sonnet류 500 prompts × 8192 입력/8192 최대 생성에서 canonical 3-run 평균 **10,956.5 tps**, vanilla **4,679.8 tps** 대비 **+134.12%**를 기록했다. fileciteturn66file0L3-L3 fileciteturn62file0L3-L3 fileciteturn64file0L3-L3 fileciteturn58file0L3-L3

하지만 이 베스트는 코딩 워크로드에 그대로 적용하면 역효과가 난다. `SUB_071` large-scale 검증에서 chat은 **+37.5%** 이득을 봤지만, code는 **−23.2%** 회귀했다. 저장소 내부 분석 문서는 이 차이를 `prompt ↔ generated` 어휘 중첩과 acceptance 차이로 설명하며, sonnet은 높고 chat은 부분적이며 code는 사실상 0에 가깝다고 정리한다. 즉, 현재 베스트는 “전역 베스트”가 아니라 **반복 어휘가 많은 sonnet 계열의 국소 베스트**다. fileciteturn67file0L3-L3 fileciteturn68file0L3-L3

코딩 워크로드 관점에서 중요한 결론은 세 가지다. 첫째, 현 브랜치의 thread-cap 패치 자체는 유효하지만, code처럼 acceptance가 거의 없는 입력에서는 `_compute_ngram_local → batch_propose_numba → _find_longest_matched_ngram_and_propose_tokens` CPU 경로가 **매 step 반복 실행되면서도 실익이 거의 없다**. 둘째, 저장소가 이미 시도한 `threshold lower`, `broadcast_object`, `speculative precompute`는 sonnet 기준 모두 noise 또는 회귀였다. 셋째, 그래서 code 최적화의 우선순위는 “스레드를 더 많이 쓰기”가 아니라 **낭비되는 CPU 병렬성을 줄이거나, code-like 요청에서 speculative path를 빠르게 끄는 것**이다. 특히 `broadcast_object` 기반 rank-0 broadcast와 full-copy precompute는 저장소가 직접 기각한 방식이므로, code 전용 개선안은 이 실패 원인을 피해야 한다. fileciteturn69file0L3-L3 fileciteturn70file0L3-L3 fileciteturn94file0L3-L3 fileciteturn95file0L3-L3 fileciteturn96file0L3-L3 fileciteturn64file0L3-L3

이 보고서의 권고는 다음과 같다. 운영상 즉시 효과가 가장 큰 선택은 **code-like prompt에서 spec OFF** 또는 **초기 수십 step에서 acceptance/empty-draft를 보고 request-local하게 spec을 끄는 동적 fallback**이다. 그다음 단계로는 `broadcast_object` 대신 **고정형 tensor/bitmap 기반의 compact no-draft propagation**, full `token_ids_cpu.copy()` 대신 **증분형 suffix index / rolling-hash negative-match fast path**, workload별 `cap/divide_by_tp` 정책 분리 같은 CPU 병렬성 개선이 적합하다. 보수적으로 보면, 순수 CPU miss-path 최적화만으로는 한 자릿수 후반 이하의 개선에 그칠 가능성이 크고, code의 **−23.2%**를 실제로 되돌리는 핵심은 여전히 **gating과 빠른 fallback**이다. 이 판단은 저장소가 `ngram lookup` 자체를 step의 1–2% 수준으로 보고, 대신 zero-acceptance code에서 전체 wall이 누적된다고 분석한 것과 정합적이다. fileciteturn67file0L3-L3 fileciteturn68file0L3-L3 fileciteturn97file0L3-L3

## 정보 요구와 조사 범위

이 질의를 제대로 답하려면 최소한 다섯 가지를 분명히 해야 했다. 첫째, CPU thread 수가 실제로 어디서 계산되고 어디서 적용되는가. 둘째, `divide_by_tp`가 per-rank thread 수를 어떻게 바꾸는가. 셋째, code workload 회귀가 acceptance 문제인지, CPU 경로 문제인지, 둘 다인지. 넷째, thread logic을 바꾼 실제 커밋과 그 이후 실패한 후속 실험이 무엇인지. 다섯째, 코드 수정 없이도 재현 가능한 베스트 구성과, code 전용 개선안의 검증 방법이 무엇인지다.

증거 우선순위는 저장소 내부의 persisted artifact에 두었다. 최상위 태스크 문서인 `TSK_020.md`, 베스트 인덱스인 `TSK_020/README.md`, canonical best 설명서인 `Best_SpecDecode_10778tps.md`, 측정 결과인 `sub044`, `sub047`, `sub071`, 그리고 실제 런타임 코드인 `vllm/v1/spec_decode/ngram_proposer.py`, `vllm/config/speculative.py`, `vllm/entrypoints/llm.py`를 주 증거로 삼았다. 추가 웹 검색은 하지 않았다. 저장소 자체가 베스트 구성, 실험 히스토리, 실패 실험의 이유, 관련 커밋까지 대부분 직접 담고 있기 때문이다. fileciteturn62file0L3-L3 fileciteturn58file0L3-L3 fileciteturn64file0L3-L3 fileciteturn65file0L3-L3 fileciteturn66file0L3-L3 fileciteturn67file0L3-L3

중요한 한계도 있다. raw launcher와 wrapper는 `/tmp/run_sub047_t3_verify_2runs.sh`, `/tmp/run_spec_decode.py`, `/tmp/run_workload_gen.py`처럼 문서에만 남아 있고 저장소에 커밋되어 있지 않다. 더구나 `de85efff` 커밋이 `eval/results/**/engine.log.stdout`, `engine.log.stderr`, profiler traces를 `.gitignore`로 제외하면서, code workload의 직접 로그를 저장소에서 재열람하기 어렵게 만들었다. 그래서 아래의 code-hotspot 분석은 **committed RESULTS/result.json + 코드 본문 + 저장소 내부 분석 문서**를 종합한 분석이며, raw engine log profiler에 의존한 line-level profile report는 아니다. fileciteturn90file0L3-L3 fileciteturn67file0L3-L3 fileciteturn68file0L3-L3

또 하나의 해석상 주의점은 커밋 메시지의 태그다. thread-cap 패치와 관련 문서 커밋은 여전히 `TSK_019` 태그를 쓰는 경우가 많지만, 현재 브랜치의 문서 구조에서는 이 spec-decode 계보가 `TSK_020`로 정리되어 있다. 따라서 아래에서는 **현재 문서 체계 기준으로는 TSK_020**, **commit lineage 기준으로는 TSK_019 spec-decode inheritance**라고 구분해 설명한다. fileciteturn62file0L3-L3 fileciteturn59file0L3-L3 fileciteturn92file0L3-L3 fileciteturn93file0L3-L3

## 파일·커밋·코드 경로 안내

### 문서와 측정 산출물

아래 표는 `feat/spec-decode-tuning`에서 TSK_020와 CPU threading, 그리고 code workload 해석에 직접 연결되는 persisted artifact를 파일 단위로 정리한 것이다.

| 파일 | 조회한 slice | 역할 | 핵심 내용 | 근거 |
|---|---|---|---|---|
| `shadow_assists/features/IDE_006/TSK_020.md` | 1–260 | 상위 태스크 문서 | TSK_020의 범위, code change 위치, best history, 관련 커밋의 정식 허브 | fileciteturn62file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/README.md` | 1–260 | 베스트 인덱스 | 현 absolute best 10,956.5, SUB_065~069 기각, plateau 선언 | fileciteturn58file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/INDEX.md` | 1–260 | navigation hub | active SUB, `SUB_071` 결과, next step 전체 맥락 | fileciteturn63file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/Best_SpecDecode_10778tps.md` | 1–320 | canonical mechanism doc | 베스트 구성, pipeline, cap=8의 이유, code workload 한계, SUB_070 현황 | fileciteturn64file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/measurements/sub044_spec_decode_20260523/RESULTS.md` | 1–240 | first net-positive | `num_speculative_tokens=7`이 sweet spot, `10` OOM | fileciteturn65file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/measurements/sub047_t3_3run_verify_20260523/RESULTS.md` | 1–260 | canonical 3-run | 베스트 구성 값, env, sampling, variance | fileciteturn66file0L3-L3 |
| `eval/results/20260523_162456_sub047_t3_verify/run2_cap8_div0/result.json` | 1–220 | machine-readable result | `method=ngram`, `num_speculative_tokens=7`, `prompt_lookup_min/max=2/5`를 JSON으로 재확인 | fileciteturn87file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/measurements/sub071_workload_large_20260524/RESULTS.md` | 1–280 | workload generalization | chat +37.5%, code −23.2%, out_tok 특성, production gating 권고 | fileciteturn67file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/analysis/workload_acceptance_analysis_20260524.md` | 1–340 | mechanism explainer | sonnet/chat/code의 K·acceptance·wall-ratio 모델, code 회귀 원인 해석, detector 제안 | fileciteturn68file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/planning/SUB_065_ngram_threshold_lower.md` | 1–220 | threshold 실험 설계 | 작은 decode batch에서 multi-thread 진입 임계값을 낮추려 한 이유 | fileciteturn94file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/planning/SUB_066_ngram_broadcast.md` | 1–240 | rank-0 broadcast 설계 | TP 8 ranks의 duplicate lookup 제거 아이디어와 리스크 | fileciteturn95file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/planning/SUB_067_speculative_ngram_precompute.md` | 1–240 | async precompute 설계 | GPU forward 동안 background precompute를 겹치려는 설계와 위험 | fileciteturn96file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/planning/SUB_070_engine_config_sweep.md` | 1–260 | post-plateau 재정의 | ngram 내부 최적화보다 GPU concurrency가 진짜 lever일 수 있다는 자체 비판 | fileciteturn97file0L3-L3 |
| `shadow_assists/features/IDE_006/TSK_020/planning/SUB_071_workload_large_chatcode.md` | 1–260 | code/chat large 설계 | code builder가 max_tokens까지 길게 생성할 수 있다는 예상과 검증 설계 | fileciteturn98file0L3-L3 |

이 file set을 보면, TSK_020에서 “코드에 손을 댄 실험”보다 “문서화된 실패 실험”이 오히려 더 중요하다. 이유는 저장소가 이미 `threshold`, `broadcast`, `precompute`, `sorting` 같은 후속 레버를 직접 시도했고, why-not까지 기록해 두었기 때문이다. 따라서 code workload용 개선안을 제시할 때는 반드시 이 실패들을 회피하도록 설계해야 한다. fileciteturn58file0L3-L3 fileciteturn63file0L3-L3 fileciteturn64file0L3-L3

### 런타임 코드 경로

아래 표는 실제 CPU thread와 speculative path를 형성하는 런타임 코드다.

| 파일 | 조회한 slice | 역할 | TSK_020/CPU threading과의 관계 | 근거 |
|---|---|---|---|---|
| `vllm/entrypoints/llm.py` | 1–260, 260–420 | Python API entrypoint | `LLM(...)` kwargs를 `EngineArgs`로 만들고 `disable_log_stats` default를 주입 | fileciteturn78file0L3-L3 fileciteturn79file0L3-L3 |
| `vllm/config/speculative.py` | 1–320, 320–620 | `SpeculativeConfig` 정의/검증 | `method="ngram"` 정규화, `prompt_lookup_{min,max}` default/validation의 source of truth | fileciteturn76file0L3-L3 fileciteturn77file0L3-L3 |
| `vllm/v1/spec_decode/ngram_proposer.py` | 1–240, 240–520, 520–820 | ngram proposer 핵심 코드 | `VLLM_NGRAM_*` env read, per-rank thread 계산, batch propose, negative match scan, top-M/broadcast/precompute path | fileciteturn69file0L3-L3 fileciteturn70file0L3-L3 fileciteturn71file0L3-L3 |
| `vllm/config/cache.py` | 1–260 | KV cache config | `gpu_memory_utilization` 기본값 0.9, `cache_dtype="auto"` 정의, FP8 KV의 의미 | fileciteturn80file0L3-L3 |
| `vllm/config/scheduler.py` | 1–260 | scheduler config | `max_num_batched_tokens` 기본 2048, `max_num_seqs` 기본 128, batched-tokens의 graph 영향 정의 | fileciteturn81file0L3-L3 |
| `vllm/sampling_params.py` | 1–260 | generation sampling | `temperature=1.0`, `top_p=1.0`, `max_tokens=16`, `seed=None` 기본값 정의 | fileciteturn82file0L3-L3 |

이 가운데 TSK_020의 “실제 승자”를 만든 파일은 단연 `ngram_proposer.py`다. 문서와 커밋은 모두 `a243e1c9f`가 upstream의 사실상 1-thread/rank 경로를 환경변수로 제어 가능한 형태로 바꿨다고 명시한다. 반대로 `llm.py`, `speculative.py`, `cache.py`, `scheduler.py`, `sampling_params.py`는 이 값들이 어디서 들어오고 어떤 default를 갖는지 설명하는 주변 파일이다. fileciteturn59file0L3-L3 fileciteturn62file0L3-L3 fileciteturn64file0L3-L3

### 관련 커밋

아래 표는 thread logic과 code-workload 해석까지 포함해 TSK_020에 실질적으로 중요한 커밋이다.

| 커밋 | 의미 | 왜 중요한가 | 근거 |
|---|---|---|---|
| `c93b88f11eeea1ec3d82fb946229526df692e291` | SUB_044 기록 | `spec=7`이 첫 net-positive였음을 공식화 | fileciteturn89file0L3-L3 |
| `a243e1c9fc6338193f2f10447c9f9e2dd65a08d5` | SUB_047 code patch | `VLLM_NGRAM_NUM_THREADS_CAP`/`VLLM_NGRAM_DIVIDE_BY_TP` 도입, thread logic 변경의 핵심 | fileciteturn59file0L3-L3 |
| `de85efff126e499aa688ed22e5635372785fa442` | eval/results 적재 | result artifact를 저장소에 남기면서도 raw engine logs를 제외 | fileciteturn90file0L3-L3 |
| `a3a930ec9094e04278bcd3643a98cf11867f2c41` | SUB_047 3-run 검증 | 베스트가 우연한 1-run이 아니라 안정적인 3-run임을 확정 | fileciteturn91file0L3-L3 |
| `202f780823ea080216ed5bd8b7d5d7338eae63e8` | mechanism 문서 확대 | default vs best, dual-path, cap=8 본질을 문서화 | fileciteturn92file0L3-L3 |
| `d11f9cf02de5653175876ee3f190c7a9d1cfc23a` | INDEX 갱신 | absolute best, dead-path 정리, next steps를 spec+CPU 쪽으로 전환 | fileciteturn93file0L3-L3 |

이 커밋 흐름을 보면, thread logic 변화는 사실상 한 번뿐이고, 이후 커밋들은 **그 변화가 베스트였음을 측정과 문서로 굳혀 나가는 과정**이다. 따라서 code workload 개선의 출발점은 “다른 hidden patch를 찾는 것”이 아니라, **현재 thread logic이 어떤 입력에서는 이득이고 어떤 입력에서는 낭비가 되는지**를 이해하는 것이다. fileciteturn59file0L3-L3 fileciteturn91file0L3-L3 fileciteturn92file0L3-L3

## 베스트 구성과 CPU 스레드 파라미터

현재 저장소에 남아 있는 베스트 구성의 정식 source of truth는 `sub047_t3_3run_verify_20260523/RESULTS.md`와 `run2_cap8_div0/result.json`이다. 여기서 추출되는 persisted truth는 Python API kwargs와 environment variables이며, **명시적 CLI flags는 저장소에 커밋된 형태로 남아 있지 않다**. 문서가 launcher 경로(`/tmp/run_sub047_t3_verify_2runs.sh`)를 언급하긴 하지만, 그 스크립트 자체는 저장소에 없으므로 아래 표에서는 **repo-backed 실제 값만** 정리했다. fileciteturn66file0L3-L3 fileciteturn87file0L3-L3 fileciteturn64file0L3-L3 fileciteturn90file0L3-L3

```python
LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=8,
    max_model_len=16384,
    max_num_seqs=256,
    gpu_memory_utilization=0.85,
    enforce_eager=False,
    kv_cache_dtype="fp8",
    max_num_batched_tokens=8192,
    disable_log_stats=True,
    seed=0,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 7,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    },
)
```

이 스니펫은 저장소 문서가 명시한 베스트 constructor 구성을 그대로 축약한 것이다. fileciteturn66file0L3-L3

```bash
export HF_HUB_OFFLINE=1
export LD_PRELOAD=/usr/lib64/libcuda.so.1
export VLLM_NGRAM_NUM_THREADS_CAP=8
export VLLM_NGRAM_DIVIDE_BY_TP=0
```

이 네 값이 베스트 실험에 같이 기록된 환경변수이며, CPU thread 관점에서 실제로 중요한 것은 뒤 두 개다. fileciteturn66file0L3-L3

### 통합 파라미터 표

| 범주 | 파라미터 | 베스트 값 | 코드 기본값 | 실제 읽는 위치 | 의미와 code workload 해석 | 근거 |
|---|---|---:|---:|---|---|---|
| 모델 | `model` | `meta-llama/Llama-3.3-70B-Instruct` | 없음 | `LLM(...)` | 측정 결과가 귀속되는 실제 타깃 모델 | fileciteturn66file0L3-L3 |
| 병렬 | `tensor_parallel_size` | `8` | `1` | `llm.py` → `EngineArgs`; `ngram_proposer.py`에서 `tp_size` 사용 | `DIVIDE_BY_TP`와 broadcast path의 기준 축 | fileciteturn78file0L3-L3 fileciteturn69file0L3-L3 |
| 길이 | `max_model_len` | `16384` | model-derived | `LLM(...)`; `NgramProposer.max_model_len` | 8192 입력 + 8192 생성 상한, proposal 길이 clipping | fileciteturn66file0L3-L3 fileciteturn69file0L3-L3 |
| 스케줄러 | `max_num_seqs` | `256` | `128` | `SchedulerConfig`; `NgramProposer` prealloc | proposer buffer shape `(max_num_seqs, k)`까지 결정 | fileciteturn81file0L3-L3 fileciteturn69file0L3-L3 |
| 캐시 | `gpu_memory_utilization` | `0.85` | `0.9` | `CacheConfig` | 현재 env에서는 0.85가 안전선. 10 speculative token은 OOM, 0.90는 별도 sweep에서 timeout 이슈 | fileciteturn80file0L3-L3 fileciteturn65file0L3-L3 fileciteturn64file0L3-L3 |
| 실행 | `enforce_eager` | `False` | `False` | `LLM(...)` | hybrid CUDA graph 경로, code path 직접 영향은 작음 | fileciteturn66file0L3-L3 fileciteturn78file0L3-L3 |
| 캐시 dtype | `kv_cache_dtype` | `fp8` | `auto` | `CacheConfig.cache_dtype` | KV footprint 절감. 정확도와 trade-off 가능 | fileciteturn80file0L3-L3 |
| 스케줄러 | `max_num_batched_tokens` | `8192` | `2048` | `SchedulerConfig` | step당 예산을 넓힘. compiled graph와 scheduler shape에 영향 | fileciteturn81file0L3-L3 |
| 로깅 | `disable_log_stats` | `True` | `LLM`에서 암묵적으로 `True` | `llm.py` | 베스트 측정은 깔끔하지만 acceptance 계측에는 불리 | fileciteturn79file0L3-L3 |
| 엔진 시드 | `seed` | `0` | `0` | `LLM(...)` | 재현성 확보 | fileciteturn66file0L3-L3 fileciteturn78file0L3-L3 |
| spec method | `speculative_config.method` | `ngram` | `None` | `SpeculativeConfig.__post_init__` | 현재 브랜치의 승자 경로를 강제 | fileciteturn77file0L3-L3 |
| spec 길이 | `num_speculative_tokens` | `7` | `None` | `SpeculativeConfig`; `NgramProposer.k` | sonnet에서는 sweet spot, 10은 OOM. code에서는 길수록 낭비가 커질 가능성 | fileciteturn65file0L3-L3 fileciteturn69file0L3-L3 |
| lookup max | `prompt_lookup_max` | `5` | 둘 다 None이면 `5` | `SpeculativeConfig`; `NgramProposer.max_n` | longest match 탐색 상한 | fileciteturn77file0L3-L3 fileciteturn69file0L3-L3 |
| lookup min | `prompt_lookup_min` | `2` | 둘 다 None이면 `5` | `SpeculativeConfig`; `NgramProposer.min_n` | 너무 짧은 match 방지. 문서 docstring과 실제 코드 default가 어긋나는 함정이 있음 | fileciteturn76file0L3-L3 fileciteturn77file0L3-L3 |
| 샘플링 | `temperature` | `0.0` | `1.0` | `SamplingParams` | greedy. 재현성 및 overlap 유지에 유리 | fileciteturn66file0L3-L3 fileciteturn82file0L3-L3 |
| 샘플링 | `top_p` | `1.0` | `1.0` | `SamplingParams` | 기본값과 동일 | fileciteturn66file0L3-L3 fileciteturn82file0L3-L3 |
| 샘플링 | `max_tokens` | `8192` | `16` | `SamplingParams` | code workload에서 긴 출력이 되며 overhead 누적이 커짐 | fileciteturn66file0L3-L3 fileciteturn67file0L3-L3 fileciteturn82file0L3-L3 |
| 런처 env | `HF_HUB_OFFLINE` | `1` | 특정 제약 없음 | launcher 수준 | 측정 재현 환경 값. CPU thread에는 직접 영향 작음 | fileciteturn66file0L3-L3 |
| 런처 env | `LD_PRELOAD` | `/usr/lib64/libcuda.so.1` | 특정 제약 없음 | launcher 수준 | 호환성/런타임 quirk로 보이며, CPU path 직접 영향은 낮음 | fileciteturn66file0L3-L3 |
| proposer env | `VLLM_NGRAM_NUM_THREADS_CAP` | `8` | `1` | `NgramProposer.__init__` | rank당 최대 numba thread cap. 베스트의 핵심 | fileciteturn69file0L3-L3 fileciteturn59file0L3-L3 |
| proposer env | `VLLM_NGRAM_DIVIDE_BY_TP` | `0` | `1` | `NgramProposer.__init__` | TP=8로 나누지 않아 rank당 8 thread 유지 | fileciteturn69file0L3-L3 fileciteturn66file0L3-L3 |
| proposer env | `VLLM_NGRAM_THRESHOLD` | 미설정 → `8192` | `8192` | `_compute_ngram_local` | multi-thread 진입 임계값. sonnet 기준 lowering은 기각 | fileciteturn69file0L3-L3 fileciteturn94file0L3-L3 fileciteturn58file0L3-L3 |
| proposer env | `VLLM_NGRAM_BROADCAST` | `0` | `0` | `batch_propose` | `broadcast_object` 기반 rank-0 only path. sonnet 기준 −1.30% | fileciteturn69file0L3-L3 fileciteturn95file0L3-L3 fileciteturn58file0L3-L3 |
| proposer env | `VLLM_NGRAM_PRECOMPUTE` | `0` | `0` | `batch_propose` + background precompute | full token copy + low hit-rate 때문에 sonnet 기준 −3.77% | fileciteturn69file0L3-L3 fileciteturn96file0L3-L3 fileciteturn58file0L3-L3 |
| proposer env | `VLLM_NGRAM_TOP_M` | `1` | `1` | `batch_propose_numba_topm` | tree expansion scaffolding. rejection sampler tree verify는 아직 TODO | fileciteturn69file0L3-L3 fileciteturn70file0L3-L3 |

현재 저장소 기준으로, CPU thread에 직접 영향을 주는 persisted knob는 사실상 `VLLM_NGRAM_NUM_THREADS_CAP`, `VLLM_NGRAM_DIVIDE_BY_TP`, `VLLM_NGRAM_THRESHOLD`, 그리고 실험용인 `VLLM_NGRAM_BROADCAST`, `VLLM_NGRAM_PRECOMPUTE`, `VLLM_NGRAM_TOP_M`이다. 반대로 사용자가 요구한 “CLI flags”는 저장소에 없으므로, 이 보고서는 CLI를 추정해 채우지 않고 **Python API kwargs + env**만 신뢰 가능한 값으로 본다. fileciteturn69file0L3-L3 fileciteturn66file0L3-L3 fileciteturn90file0L3-L3

## 실행 흐름과 파라미터 사용처

런타임 흐름은 `LLM(...)`의 kwargs에서 시작한다. `llm.py`는 `disable_log_stats`를 기본 `True`로 보정한 뒤, `EngineArgs`를 만들고 `LLMEngine.from_engine_args(...)`를 호출한다. 그 뒤 `SpeculativeConfig.__post_init__`가 `method="ngram"`을 정규화하고 `prompt_lookup_min/max`를 보정·검증한다. 마지막으로 worker 쪽 `NgramProposer`가 이 설정을 읽어 env와 함께 실제 CPU thread 계산 및 draft generation을 수행한다. rejection sampler 단계는 best mechanism 문서가 명시적으로 포함하지만, 그 concrete file은 이번 저장소 스캔 범위에서 persisted code artifact로 따로 식별되지 않았다. fileciteturn79file0L3-L3 fileciteturn77file0L3-L3 fileciteturn69file0L3-L3 fileciteturn64file0L3-L3

```python
if "disable_log_stats" not in kwargs:
    kwargs["disable_log_stats"] = True

engine_args = EngineArgs(
    model=model,
    tensor_parallel_size=tensor_parallel_size,
    seed=seed,
    gpu_memory_utilization=gpu_memory_utilization,
    enforce_eager=enforce_eager,
    **kwargs,
)
self.llm_engine = LLMEngine.from_engine_args(...)
```

이 부분에서 베스트 구성의 constructor 값들이 엔진 레벨 설정으로 넘어간다. `disable_log_stats=True`가 기본으로 들어간다는 점은 디버깅 시 특히 중요하다. fileciteturn79file0L3-L3

```python
if self.method in ("ngram", "ngram_gpu"):
    if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
        self.prompt_lookup_min = 5
        self.prompt_lookup_max = 5
    elif self.prompt_lookup_min is None:
        self.prompt_lookup_min = self.prompt_lookup_max
    elif self.prompt_lookup_max is None:
        self.prompt_lookup_max = self.prompt_lookup_min
```

이 코드는 `prompt_lookup_min/max`의 실제 default truth가 docstring보다 `__post_init__` 쪽에 있다는 점을 보여 준다. 둘 다 비어 있으면 둘 다 `5`가 되고, 현재 베스트는 이를 `2/5`로 명시적으로 override한다. fileciteturn76file0L3-L3 fileciteturn77file0L3-L3

```python
self.num_tokens_threshold = int(os.environ.get("VLLM_NGRAM_THRESHOLD", "8192"))
cap = int(os.environ.get("VLLM_NGRAM_NUM_THREADS_CAP", "1"))
divide_by_tp = int(os.environ.get("VLLM_NGRAM_DIVIDE_BY_TP", "1"))

if cpu_count:
    self.num_numba_thread_available = max(1, min(cap, (cpu_count // 2)))
    if divide_by_tp:
        self.num_numba_thread_available //= tp_size
    self.num_numba_thread_available = max(1, self.num_numba_thread_available)
```

이 초기화부가 SUB_047의 핵심이다. `cap=8, divide_by_tp=0`이면 TP=8이어도 rank당 8 thread를 유지하고, `cap=1, divide_by_tp=1`이면 upstream-like 1-thread/rank에 가깝게 동작한다. fileciteturn69file0L3-L3 fileciteturn59file0L3-L3

```python
if total_tokens >= self.num_tokens_threshold:
    final_num_threads = max(
        1, min(self.num_numba_thread_available, num_ngram_requests)
    )
    set_num_threads(final_num_threads)
else:
    set_num_threads(1)

batch_propose_numba(...)
```

thread 수를 “결정해서 실제로 쓰는” 곳은 `_compute_ngram_local()`이다. `VLLM_NGRAM_THRESHOLD`가 여기서 single-thread fallback 여부를 결정한다. fileciteturn70file0L3-L3 fileciteturn94file0L3-L3

```python
for i in prange(len(valid_ngram_requests)):
    idx = valid_ngram_requests[i]
    context_token_ids = token_ids_cpu[idx, :num_tokens_no_spec[idx]]
    drafter_output = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=context_token_ids,
        min_ngram=min_n,
        max_ngram=max_n,
        max_model_len=max_model_len,
        k=k,
    )
```

실제 CPU work의 핵심 루프는 여기다. request별로 `prange` 병렬화된 뒤, 각 request는 `_find_longest_matched_ngram_and_propose_tokens()` 안에서 suffix-match를 찾는다. code workload는 바로 이 루프가 “거의 항상 빈 draft를 돌려주는데도 계속 비싼 scan을 반복한다”는 점이 문제다. fileciteturn70file0L3-L3 fileciteturn71file0L3-L3 fileciteturn68file0L3-L3

```mermaid
flowchart TD
    A[Python API LLM(...)] --> B[llm.py: EngineArgs 생성]
    B --> C[LLMEngine.from_engine_args]
    C --> D[SpeculativeConfig.__post_init__]
    D --> E[method=ngram, lookup 2/5 확정]
    E --> F[NgramProposer.__init__]
    F --> G[env 읽기: CAP DIVIDE_BY_TP THRESHOLD BROADCAST PRECOMPUTE TOP_M]
    G --> H[per-rank numba thread 수 계산]
    H --> I[propose]
    I --> J[batch_propose]
    J --> K[_compute_ngram_local]
    K --> L[set_num_threads 또는 single-thread fallback]
    L --> M[batch_propose_numba]
    M --> N[_find_longest_matched_ngram_and_propose_tokens]
    N --> O[draft 후보 반환]
    O --> P[GPU verify 1+7 tokens]
    P --> Q[rejection sampler]
    Q --> R[accept된 prefix 반영]
    R --> S[EOS 또는 max_tokens까지 반복]
```

이 흐름에서 CPU thread 관련 값을 요청당 동작에 연결해 보면, `VLLM_NGRAM_NUM_THREADS_CAP`과 `VLLM_NGRAM_DIVIDE_BY_TP`는 **“몇 thread로 propose kernel을 돌릴지”**를 결정하고, `VLLM_NGRAM_THRESHOLD`는 **“언제 multi-thread를 켤지”**를 결정한다. `prompt_lookup_min/max`와 `num_speculative_tokens`는 **“CPU가 무엇을 얼마나 깊게 찾을지”**를 결정하며, code workload에서는 바로 이 탐색 비용이 거의 성과 없이 반복되는 쪽으로 나타난다. fileciteturn69file0L3-L3 fileciteturn70file0L3-L3 fileciteturn71file0L3-L3 fileciteturn68file0L3-L3

## 코딩 워크로드 회귀와 CPU 병목

가장 직접적인 사실부터 보면, large 500p × 8192 × 8192 조건에서 code workload는 vanilla **6,964.5 tps**, spec7+cap8 **5,346.8 tps**로 **−23.2%** 회귀했다. 입력 토큰 수는 같고 출력 토큰 수도 거의 동일하다. 그런데 wall은 vanilla **562.10s**에서 spec **726.23s**로 늘었다. 이는 “같은 양의 토큰을 생성했는데 speculative path의 추가 비용만 누적되었다”는 뜻이다. 문서도 code 응답이 평균 약 **7,830 tokens/prompt**로 거의 `max_tokens`까지 길게 생성되며, ngram acceptance가 매우 낮다고 해석한다. fileciteturn67file0L3-L3 fileciteturn68file0L3-L3

저장소의 acceptance analysis는 이를 정량 모델로 풀어 쓴다. sonnet은 `K≈3.0~5.0`, chat은 `K≈1.7~1.9`, code는 `K≈1.0`으로 요약된다. 여기서 `K`는 speculative step당 실제 진척 token 수이고, code가 `K≈1.0`이라는 말은 acceptance가 사실상 0이라서 **step compression 효과가 없다는 뜻**이다. 이런 상황에서는 `1+7` token verify, lookup, rejection-sampling 분기, rollback 메타데이터 비용이 그대로 wall에 노출된다. fileciteturn68file0L3-L3

code workload의 CPU 관점 병목을 코드 경로에 맞춰 옮겨 적으면 아래와 같다.

| hotspot | code에서 왜 문제인가 | 판단 근거 |
|---|---|---|
| `_compute_ngram_local` | step마다 thread 수를 세팅하고 kernel 호출 경로를 밟는다. acceptance가 거의 0이어도 이 진입 비용은 계속 발생한다. | `_compute_ngram_local`가 매 propose마다 호출되고, code는 장출력이라 step 수가 매우 많다. fileciteturn70file0L3-L3 fileciteturn67file0L3-L3 |
| `batch_propose_numba` | `prange`로 request별 draft를 찾지만, code에서는 실제로 empty draft가 지배적일 가능성이 높아 “병렬 empty search”가 된다. | code의 acceptance≈0, K≈1.0 해석과 request별 prange 구조를 결합한 추론이다. fileciteturn68file0L3-L3 fileciteturn70file0L3-L3 |
| `_find_longest_matched_ngram_and_propose_tokens` | 매 step 전체 context suffix를 뒤집어 longest match를 찾지만, code prompt의 word-salad와 실제 생성되는 Python 코드 분포가 맞지 않아 빈 결과가 잦다. | 분석 문서가 code prompt와 generated code의 어휘 분포가 거의 disjoint라고 설명한다. fileciteturn68file0L3-L3 fileciteturn71file0L3-L3 |
| rejection sampler | CPU가 아니더라도 제어 흐름상 매 step 추가 분기·검증·rollback 경로를 만든다. acceptance가 0에 가깝다면 이 오버헤드는 전부 순손실이다. | best mechanism 문서와 analysis 문서가 rejection sampler를 spec path의 필수 단계로 설명한다. fileciteturn64file0L3-L3 fileciteturn68file0L3-L3 |

여기서 특히 중요한 점은 “CPU miss-path만 빨리 만들어도 code 회귀가 전부 없어지지는 않는다”는 사실이다. 저장소는 sonnet 기준으로 `ngram time ~1–2 ms`, step time `70–90 ms`라고 자체 비판했고, 그래서 `SUB_065~069`가 모두 plateau/noise였다고 결론냈다. 즉, 순수 lookup 비용 최적화만으로는 end-to-end 이득 상한이 크지 않다. code workload에서 CPU miss-path가 더 많이 반복되므로 체감 이득은 sonnet보다 커질 수 있겠지만, **−23.2% 전체 회귀를 되돌리는 1순위 해법은 여전히 speculative path 자체를 빨리 끄는 것**이다. fileciteturn97file0L3-L3 fileciteturn58file0L3-L3 fileciteturn67file0L3-L3

저장소가 이미 해본 실패 실험들도 이 해석을 강하게 뒷받침한다. `VLLM_NGRAM_THRESHOLD`를 낮춰 작은 batch에서도 multi-thread를 강제로 켜려던 `SUB_065`는 모두 baseline 동등 혹은 미세 회귀였고, rank-0만 계산 후 `broadcast_object`를 하던 `SUB_066`은 pickle/broadcast overhead 때문에 **−1.30%**, background precompute를 붙인 `SUB_067`은 `token_ids_cpu.copy()`와 low hit-rate 때문에 **−3.77%**였다. 따라서 code 전용 CPU 병렬성 최적화는 **blind multithreading**, **generic object broadcast**, **full-copy speculative precompute**를 그대로 반복하면 안 된다. fileciteturn94file0L3-L3 fileciteturn95file0L3-L3 fileciteturn96file0L3-L3 fileciteturn58file0L3-L3

## 코딩 워크로드용 개선안

아래 제안은 모두 저장소의 현재 베스트와 실패 실험을 전제로 한 것이다. 즉, **왜 sonnet에서 안 먹혔는지**, **왜 code에서는 오히려 더 맞을 수 있는지**, **어떤 실패 원인을 피하는지**를 함께 적었다. 기대 효과는 저장소 측정값을 바탕으로 한 **보수적 추론**이다.

### 즉시 적용 가능한 운영 개선

| 제안 | 구현 포인트 | code에 특히 맞는 이유 | 기대 효과 | 근거 |
|---|---|---|---|---|
| **workload-aware spec OFF 라우팅** | prompt detector로 code-like 요청을 vanilla instance로 보냄 | 저장소 자체가 code는 −23.2%, chat은 +37.5%, sonnet은 +134.1%라고 확인 | 가장 큰 효과. code 회귀 대부분을 바로 제거 가능 | fileciteturn67file0L3-L3 fileciteturn68file0L3-L3 |
| **초기 decode 단계의 request-local fallback** | 첫 16–32 step 동안 empty-draft ratio 또는 acceptance를 보고 개별 request만 spec OFF 전환 | prompt만 보고 분류하는 것보다 안전하고, code처럼 긴 출력에서는 초반 판단만으로 대부분의 남은 step을 절약 | 운영 혼합 트래픽에서 spec-on의 장점은 유지하면서 code drag를 크게 줄일 가능성 | fileciteturn67file0L3-L3 fileciteturn68file0L3-L3 fileciteturn79file0L3-L3 |
| **code bucket 전용 thread downshift** | code-like 요청은 `cap=1~2`, `divide_by_tp=1` 또는 완전 single-thread 유지 | acceptance가 0이면 “더 많은 thread로 더 빨리 틀린 draft를 찾는 것” 자체가 낭비 | end-to-end 상한은 크지 않지만 CPU 낭비와 contention을 줄일 수 있음 | fileciteturn69file0L3-L3 fileciteturn67file0L3-L3 fileciteturn68file0L3-L3 |

이 셋 중 우선순위는 분명하다. **가장 먼저 할 일은 code-like 요청을 speculative path에서 빼는 것**이다. 저장소 분석 문서도 detector heuristic으로 `def`, `class`, triple backticks, `import`, template tags, unique-token ratio를 제안하고 있고, 이를 production gating의 자연스러운 다음 step으로 본다. fileciteturn68file0L3-L3

### 코드 수정이 필요한 CPU 병렬성 개선

| 제안 | 어디를 바꾸는가 | 기존 실패 실험과 다른 점 | 기대 효과 | 근거 |
|---|---|---|---|---|
| **negative-match fast path** | `NgramProposer`에 per-request suffix hash 또는 terminal-token posting list 추가 | `SUB_067`처럼 full `token_ids_cpu.copy()`를 하지 않고, acceptance 추정에도 의존하지 않음 | code에서 empty draft가 지배적일 때 `_find_longest...` 진입 자체를 자주 생략 가능. 보수적으로는 소~중간 개선 | fileciteturn71file0L3-L3 fileciteturn68file0L3-L3 fileciteturn96file0L3-L3 |
| **증분형 suffix index 유지** | 각 request에 대해 “현재 suffix → 이전 occurrence” 인덱스를 한번 만들고 step마다 O(1)/O(log n) 업데이트 | 매 step 전체 context를 다시 스캔하는 현재 구조를 줄이는 방향이며, speculative precompute처럼 다음 acceptance를 가정하지 않음 | code처럼 7.8k tok/prompt의 긴 출력을 다루는 데 적합. miss-path 비용을 많이 누적시키는 입력에서 더 유리 | fileciteturn67file0L3-L3 fileciteturn70file0L3-L3 fileciteturn71file0L3-L3 |
| **compact tensor broadcast of no-draft bitmap** | `broadcast_object` 대신 preallocated `torch.int32`/`torch.uint8` tensor 사용, 혹은 “초안 없음” bitmask만 broadcast | `SUB_066`이 실패한 이유는 object pickle + CPU group broadcast가 비싼 것이었지, duplicate 제거 자체가 틀렸다는 뜻은 아님 | code는 draft가 자주 비므로 payload가 작다. sonnet보다 code에서 더 유리할 가능성이 있음 | fileciteturn95file0L3-L3 fileciteturn58file0L3-L3 fileciteturn67file0L3-L3 |
| **code bucket 전용 no-parfor kernel** | `batch_propose_numba`의 code-like/empty-draft 우세 batch에서 `prange` 대신 단순 `njit` fast-negative kernel 사용 | `SUB_065`처럼 무조건 멀티스레드를 켜는 방향과 반대다. code에서는 “병렬성 확대”보다 “낭비 제거”가 핵심 | CPU 시간을 아끼고 oversubscription을 줄임. 단독 효과는 제한적일 가능성이 크다 | fileciteturn94file0L3-L3 fileciteturn97file0L3-L3 |

이 제안군의 공통점은 저장소가 이미 실패한 세 가지 패턴을 피한다는 점이다. **object broadcast를 피하고**, **full token copy를 피하고**, **작은 batch에서 멀티스레드를 무조건 켜지 않는다**. 특히 `compact no-draft bitmap`은 `SUB_066`의 core idea를 버리는 게 아니라, 실패 원인인 Python object serialization을 제거해 code workload에 맞게 다시 설계하자는 제안이다. fileciteturn95file0L3-L3 fileciteturn96file0L3-L3 fileciteturn94file0L3-L3

### 기대 효과의 현실적 상한

이 개선안들의 기대 효과는 과장하면 안 된다. 저장소의 자체 비판대로 sonnet에서는 ngram lookup이 전체 step의 1–2%에 불과했고, 그래서 lookup 내부만 만지는 `SUB_065~069`가 모두 plateau였다. code에서는 empty-draft가 길게 반복되므로 miss-path 최적화의 절대 효과가 sonnet보다 커질 수는 있어도, lookup 최적화만으로 `−23.2%` 전체를 한 번에 없애기는 어렵다. 따라서 **가장 큰 효과는 gating/early fallback**, 그다음이 **negative fast path + compact broadcast**, 마지막이 **세부 thread tuning**이라는 우선순위가 타당하다. fileciteturn97file0L3-L3 fileciteturn67file0L3-L3 fileciteturn68file0L3-L3

## 검증 전략과 운영 주의점

가장 먼저 해야 할 검증은 acceptance를 직접 보이게 만드는 것이다. 현재 베스트는 `disable_log_stats=True`라서 내부 acceptance metric이 잘 보이지 않는다. 저장소 분석 문서는 `disable_log_stats=False`로 바꾸면 `num_draft_tokens`, `num_accepted_tokens`, `acceptance_rate` 같은 metric을 로그에서 직접 뽑을 수 있다고 제안한다. code workload 개선은 거의 전부 “acceptance가 낮을 때 빨리 빠져나가자”는 아이디어이므로, 이 계측이 먼저 없으면 최적화가 아니라 추측에 가까워진다. fileciteturn79file0L3-L3 fileciteturn68file0L3-L3

테스트는 네 층으로 나누는 것이 적절하다. 첫째, **단위 테스트**로 thread 수 유도식이 맞는지 확인한다. `cap=8/divide_by_tp=0/tp=8`이면 8, `cap=8/divide_by_tp=1/tp=8`이면 1이 나와야 한다. 둘째, **microbenchmark**로 `_find_longest_matched_ngram_and_propose_tokens()`의 no-match dominant code prompt를 따로 재현한다. 셋째, **canonical benchmark**로 sonnet 3-run과 code/chat 비교 실험을 돌려 baseline drift를 분리한다. 넷째, **guard test**로 `spec=10` OOM, `broadcast_object` 회귀, `precompute` full-copy 회귀 같은 이미 알려진 실패를 다시 기본값에 섞지 않도록 막는다. fileciteturn65file0L3-L3 fileciteturn66file0L3-L3 fileciteturn67file0L3-L3 fileciteturn95file0L3-L3 fileciteturn96file0L3-L3

운영상 주의점도 분명하다. `cap=56`은 `SUB_047`에서 **−17%** 회귀였고, multi-SUB combo 역시 NUMA contention으로 **−6% ~ −12%** 회귀했다. 즉 “CPU를 더 쓰자”가 항상 좋은 게 아니다. code workload에서 acceptance가 낮을 때는 특히 **더 많은 thread가 더 많은 낭비**가 될 수 있다. 따라서 code 최적화는 “스레드 수를 키우는 것”보다 “틀린 speculative work를 안 하게 만드는 것”이 우선이다. fileciteturn62file0L3-L3 fileciteturn58file0L3-L3 fileciteturn63file0L3-L3

의존성 측면에서 CPU path에 직접 중요한 버전은 아래와 같다.

| 항목 | 저장소 요구사항 | CPU path와의 관련성 | 근거 |
|---|---|---|---|
| Python | `>=3.10,<3.14` | 런타임 지원 범위 | fileciteturn83file0L3-L3 |
| `torch` | `2.11.0` | 전체 런타임 및 build 기준 | fileciteturn83file0L3-L3 fileciteturn85file0L3-L3 fileciteturn86file0L3-L3 |
| `numba` | `0.61.2` | ngram speculative decoding 필수 | fileciteturn85file0L3-L3 fileciteturn86file0L3-L3 |
| `numpy` | 버전 고정 없음 | proposer buffer와 numba kernel 입력 | fileciteturn84file0L3-L3 fileciteturn69file0L3-L3 |
| `transformers` | `>=4.56.0` | 모델/토크나이저 config 계층 | fileciteturn84file0L3-L3 |
| `tokenizers` | `>=0.21.1` | fast incremental detokenization | fileciteturn84file0L3-L3 |
| `pydantic` | `>=2.12.0` | config validation (`SpeculativeConfig`, `CacheConfig`, `SchedulerConfig`) | fileciteturn84file0L3-L3 |
| CUDA 쪽 보조 | `flashinfer-python==0.6.7` 등 | GPU verify path 간접 관련 | fileciteturn86file0L3-L3 |

반면, OS 배포판, 정확한 CUDA driver 버전, libc, container image 태그, HF model revision, launcher shell 옵션, profiler 설정 등은 저장소의 TSK_020 artifact에 **명시적으로 고정되어 있지 않다**. 사용자의 조건대로 이런 항목은 이 보고서에서 **“no specific constraint”**로 본다. 다만 측정 환경 자체는 문서가 반복해서 `H100×8`, `TP=8`, `FP8 KV`, `Llama-3.3-70B`라고 밝히므로, 이것은 “필수 버전”이 아니라 **실측 기준 환경**으로 읽는 것이 맞다. fileciteturn62file0L3-L3 fileciteturn66file0L3-L3 fileciteturn64file0L3-L3

최종적으로, `feat/spec-decode-tuning`의 TSK_020는 sonnet류에서는 매우 강한 베스트를 갖고 있지만, code workload에는 그 베스트를 그대로 적용하면 안 된다. code용 CPU 병렬성 최적화의 정답은 “더 많은 병렬성”이 아니라 **낭비되는 speculative 병렬성을 빨리 꺼 버리는 것**, 그리고 정말 CPU 쪽을 건드려야 한다면 **negative-match fast path**, **compact no-draft propagation**, **incremental index** 같은 방식으로 실패한 SUB_065/066/067의 비용 구조를 피하는 것이다. fileciteturn67file0L3-L3 fileciteturn68file0L3-L3 fileciteturn95file0L3-L3 fileciteturn96file0L3-L3
