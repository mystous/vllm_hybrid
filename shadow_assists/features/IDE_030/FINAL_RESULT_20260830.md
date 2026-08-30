# 480B 하이브리드 서빙 최종 결과 (2026-08-30)

## 한 줄 요약

Qwen3-Coder-480B(FP8, 450GB)를 **GPU 4장 + CPU로 서빙하면서 56.3 → 324 tok/s (+475%)**. 품질 저하 없음(GSM8K 85.0% → 95.0%).

> 갱신 (16:30): 부분 CUDA graph(`tc_piecewise`) + hot-80 으로 152.6 → 229 → 324 (GSM8K 97.5%). graph가 GPU를 빠르게 하자 최적 hot 개수가 64→80으로 이동. rank 간 동기화 대기(프로파일 실측 47%)가 제거된 효과. 이 구성에서 초안 검증은 제외(graph와 결합 시 충돌 — 향후 과제). 상세는 `eval/results/*_pln007_spec480/RESULTS.md`의 CUDA graph 트랙 절.

## 최종 구성

| 항목 | 값 |
|---|---|
| 모델 | Qwen3-Coder-480B-A35B-Instruct-FP8 (expert 층 62개 × 160개) |
| GPU | H100 4장 (TP=4) — attention·KV·**호출 빈도 상위 expert 64개/층 (FP8)** |
| CPU | Xeon 8480+ ×2, 96스레드, threadpool 2 — **나머지 expert 96개/층 (INT4, DRAM 상주, AMX 계산)** |
| 초안 검증 | Qwen3-4B 초안 모델, K=3 (speculative decoding, 채택률 0.65) |
| 핵심 플래그 | `--kt-cpuinfer 96 --kt-threadpool-count 2 --kt-num-gpu-experts 64 --init-expert-location hotmap.json --ep-dispatch-algorithm dynamic --speculative-algorithm STANDALONE --speculative-draft-model-path <Qwen3-4B> --speculative-num-steps 3 --speculative-num-draft-tokens 4` |

hotmap.json = 라우팅 트레이스(sonnet 워크로드, 층별 expert 호출 빈도)로 만든 배치표 — 빈도 상위 expert가 물리 id 0부터 오도록 재배열. 가중치 파일 수정 없음.

## 성능 (sonnet 512/128, seed 42, CPU turbo OFF 2.0GHz)

### GPU 상주 expert 개수별 (동시 32, 초안 검증 포함)

| 개수 | 호출 커버리지 | tok/s | TPOT |
|---:|---:|---:|---:|
| 0 (기준) | 0% | 56.3 | 436ms |
| 16 | 43% | 92.4 | 277ms |
| 32 | 60% | 116.2 (3회 평균) | 217ms |
| **64** | **79%** | **152.6** | **177ms** |
| 80 | 84% | 148.9 | 175ms (수확 체감) |

### 동시성별 (hot-32 구성 시점 측정; hot-64는 C=32만 측정)

| 동시성 | 기준 | 개선 | 이득 |
|---:|---:|---:|---:|
| 16 | 55.5 | 72.6 | +31% |
| 32 | 56.3 | 152.6 (hot-64) | **+171%** |
| 64 | 84.2 | 151.3 (hot-32) | +80% |

## 품질 검증

1. greedy 고정 문항 4종: 정상
2. 같은 문장에 대한 토큰별 확률 비교: 평균 차이 0.106 (작음), 최대 3.29 — 분포가 달라짐을 확인
3. **정답률 판정 (GSM8K 40문항, chat 형식, greedy)**: 기준 85.0% (34/40) → 개선 **95.0% (38/40)** — 저하 없음
   - 달라진 방향의 해석: GPU에 올라간 expert(호출의 79%)는 INT4 압축본 대신 FP8 원본 가중치로 계산되므로 원본 모델에 가까워짐

## 왜 빨라졌나 (측정으로 확인한 인과)

- 프로파일 분해: 개선 전 시간의 대부분은 CPU가 expert 가중치를 DRAM에서 읽는 시간 (GPU 가동률 1.3%). 읽기 속도는 이미 하드웨어 한계의 ~72% → "더 빨리"가 아니라 **"덜 읽게"** 해야 했음
- 호출 빈도가 매우 편중됨 (상위 64개가 호출의 79%) → 이 64개를 GPU에 두면 CPU 읽기량이 1/5로 감소
- 초안 검증은 한 번의 CPU 호출당 처리 토큰 수를 늘려 남은 CPU 비용도 상각

## 성립의 전제가 된 버그 수리 3건 (이것 없이는 위 기법이 모두 무효/오작동)

| # | 위치 | 증상 | 수리 |
|---|---|---|---|
| 1 | SGLang `qwen3_moe.forward_normal` | expert 재배치 정보가 라우터에 전달 안 됨 → 재배열 시 조용한 출력 붕괴 | 1줄 패치 (전달 추가) |
| 2 | SGLang `kt_ep_wrapper` | kt는 추론 모드에서 `gpu_experts_mask` 목록만 보는데 None으로 고정돼 있어 **CPU 전량 계산 + GPU 중복 덧셈** | 목록 생성·전달 패치 |
| 3 | kt-kernel | threadpool 수 ≠ 가중치 변환 시 `--numa-nodes` 이면 오류 없이 출력만 붕괴 | 구성 규칙 준수 (threadpool 2 고정) + upstream 제보 초안 |

(+ 기존 발견: kt가 CPU 스레드를 절대 0번 코어부터 고정하는 결함 — 다중 인스턴스 시 cgroup cpuset 격리 필수)

## 재현

1. 컨테이너 준비: `shadow_assists/features/IDE_023/scripts/setup_kt_container.sh <이름>` (호환 패치 + 위 수리 1·2 포함)
2. 라우팅 트레이스: 서버를 `--expert-distribution-recorder-mode per_pass`로 띄워 대표 워크로드 실행 → 층별 빈도 → hotmap.json 생성 (physical_to_logical_map = 층별 빈도 내림차순 정렬)
3. 위 "핵심 플래그"로 기동, 품질 문항 확인 후 측정
4. 원시 데이터: `eval/results/20260830_105328_pln007_spec480/` (전 셀 bench.log·품질·GSM 결과)

## 한계·주의

- 모든 수치는 turbo OFF (2.0GHz 고정) 하한값, 머신 1대(violet-h100-016) 실측
- hotmap은 워크로드(sonnet) 빈도 기준 — 워크로드가 다르면 빈도 재측정 권장 (층별 상위 집중은 구조적이라 대체로 이식될 것으로 추정, 미검증)
- 동시 16/64에서의 hot-64 수치는 미측정 (hot-32 값으로 하한 추정)
- **연구 신규성 없음**: 구성 요소(hot 배치·빈도 재배치·초안 검증)는 모두 기존 기법. 기여는 엔지니어링(버그 수리 + 검증된 조합)임
