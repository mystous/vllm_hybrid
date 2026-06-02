# PRE-TSK042/TSK043 — 사용자 선행조건 (Claude 불가 항목만)

> **대상**: IDE_022 의 TSK_042(워크로드 실험 + llm-d 라우팅) / TSK_043(AGSD CPU 병렬성 최적화).
> **원칙**: 본 문서에는 **Claude 가 스스로 할 수 없는 것**(외부 계정·웹 동의·인프라 권한)만 둔다. 그 외(의존 설치·모델 다운로드·빌드 검증·하니스)는 Claude 가 자동 처리.
> **형식**: CLI 에서 그대로 복사·붙여넣기. 변수 없음(리터럴 경로).

---

## ⚠ 사용자가 직접 해야 하는 것

### 1. ✅ 완료 — gated 데이터셋 라이선스 동의
`mystous` 계정으로 **LMSYS-Chat-1M, WildChat-1M 모두 동의 완료**(2026-06-02 검증, streaming 접근 OK). 추가 사용자 조치 없음.

**데이터셋 접근 검증 결과** (2026-06-02):
| corpus | 접근 | 비고 |
|---|---|---|
| ShareGPT52K (RyokoAI) | ✅ | |
| SWE-bench Lite (princeton-nlp) | ✅ | |
| WildChat-1M (allenai, gated) | ✅ | |
| LMSYS-Chat-1M (lmsys, gated) | ✅ | 동의 완료 |
| ~~LiveCodeBench~~ | ❌ | datasets 4.x script 폐지 → **HumanEval/MBPP 로 대체** (datasets-native, pass@k 는 lm-eval 내장) |

> **다운로드 정책**: LMSYS/WildChat 은 각 100만 대화 → **전체 캐시 안 함**. corpus_loader 가 corpus당 500~2000 만 **streaming 샘플링**(디스크 절약, 모델 다운로드와 공존).

### 2. llm-d k8s 인프라 결정 (TSK_042 라우팅 — 인프라 권한)
이 컨테이너엔 docker/k8s 가 없음(`minikube/kubectl/helm/docker` ❌). llm-d 는 k8s + GPU passthrough 가 필요해 **컨테이너 안에서 Claude 가 신뢰성 있게 못 띄움** → 다음 중 택1 (사용자/인프라 결정):
- (a) **호스트/별도 노드에 Minikube** 구축 후 llm-d-deployer 실행:
```bash
minikube start --driver=docker --container-runtime=docker --gpus all
git clone https://github.com/llm-d/llm-d-deployer.git && cd llm-d-deployer/quickstart && ./llmd-installer.sh
kubectl get pods && ./test-request.sh
```
- (b) llm-d 비교를 **보류**하고 TSK_042 의 oracle/품질 + vanilla/trident 단일백엔드 비교만 먼저 진행.
→ 어느 쪽인지 알려주시면 그에 맞춰 TSK_042 진행.

### 3. 모델 다운로드 (사용자 직접 — 별도 셸)
캐시됨(✅): Qwen2.5-7B/32B/72B, Llama-3.1-8B/70B. 아래는 직접 다운로드:
```bash
HF=/workspace/vllm_dev_prj/bin/hf
# T1~T3 DeepSeek distill (open, ~220GB)
$HF download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
$HF download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
$HF download deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# T4 XL(>70B, fp8) — 디스크 874GB라 순차(받고→측정→정리→다음)
$HF download meta-llama/Llama-3.1-405B-Instruct-FP8     # ~405GB
#   rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-405B-Instruct-FP8  # 측정 후
$HF download deepseek-ai/DeepSeek-R1                     # 671B fp8 ~671GB (405B 정리 후)
```

---

## ✅ Claude 가 자동 처리 (참고 — 선행조건 아님)
다음은 Claude 가 실행하므로 사용자 조치 불필요:
- 의존 설치: `uv pip install --python /workspace/vllm_dev_prj/bin/python arctic-inference==0.1.1 datasets langdetect pyarrow lm-eval google-re2 scikit-learn onnxruntime "optimum[onnxruntime]" sentence-transformers`
- mimalloc: `apt-get install -y libmimalloc2` (root 가능)
- (모델 다운로드는 §3 으로 이동 — 사용자 직접)
- sm_100 빌드 검증 / 서빙 하니스(`vllm_config_perf/gating/`) 확인
- open 데이터셋 로드(ShareGPT/LiveCodeBench/SWE-bench)

## 진입 게이트
- **TSK_042 시작**: 위 §1·§2 결정 불필요 — open corpus + Claude 자동셋업으로 oracle/품질 즉시 가능. gated(§1)·llm-d(§2)는 결정되는 대로 합류.
- **TSK_043 분류기·latency**: 즉시 독립. regret 측정만 TSK_042 의 `oracle_table.parquet` 합류.
