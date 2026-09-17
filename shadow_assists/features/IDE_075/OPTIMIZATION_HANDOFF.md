# OPTIMIZATION_HANDOFF — IDE_075 → 다음 성능 개선 단계 (이번 단계에서 성능 변경 실행 없음)

## 1. 재현 기준
- 모델 `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` rev 003f183a…, CPU 가중치 `/models/kt/qwen3-480b-int4` AMXINT4, hotmap_v2 (SHA 7ce02cad…), layer_budget_5952 (SHA c8ba4247…, 62층 합 5,952), TP4 GPU 0~3, CPU 96 workers/NUMA pool 2, deferred 8, callback-free/skip-empty/RB ON, KV 40,960, decode graph ≤64, prefill graph off, chunk 8,192, mem 0.95, triton attention. launch argv: `evidence/environment.json: reference_launch_cmd`.
- 런타임: SGLang 71de97b + 로컬 7파일, kt_kernel_ext.so **v2 a4e9045a…** (OFF/ON 공통; OFF 는 env KT_EVT 미설정). 원본 e29357f7… / v1 adf49e48… 은 `/sgl-workspace/ide074_backup/`. 복원: `python3 /tmp/ide075_kt_evt_patch_v2.py revert` + `.so.orig` 복사.
- 입력: `M073_QWEN_MAIN_SHORT` (C64, n256, 131,786/32,768 tok), `M073_QWEN_PROBE128` (C64, n128, 65,886/16,384), frozen jsonl `~/.cache/huggingface/kt/ide073/inputs/qwen/` (SHA `IDE_074 evidence/environment.json: input_files`).
- 다음 성능 비교 프로토콜: **새 MAIN_SHORT OFF 3회(같은 부팅) 를 분모로** 새로 측정 (IDE_074 의 743/747/769 는 v1 바이너리·역사값). 후보도 같은 부팅 배치·smoke·warmup·cache flush 로 3회. 계측 OFF 증빙 = requested_config.env 에 KT_EVT 없음 + task_count 기록 없음.

## 2. 확인된 지연 경로 (C64 디코드 bs=63, S3/S4 CORR, cold_present_nonempty)
```
go(L−1) → enqueue(+5 µs) → [직전 층 deferred 실행 중: 650~663 µs] → deferred(L−1) 서비스 902~912 µs (numa0/1 병렬, up_gate 55 % · down 34 %, 전 expert AVX vec_mul, rows 1~5)
   → done 게시 (+0.5) → GPU: hot_ready(L) 는 go(L−1) 로부터 1,248~1,270 µs 에 도착 → 게시가 hot 보다 315~321 µs 늦음 (90 %) → wait 해제 → HtoD 32 µs → combine +4~9
```
GPU 는 그 315~334 µs 동안 실제 idle (다른 스트림 작업 0). 생산층별 서비스는 unique cold expert 수에 비례 (≈65 µs/expert). 대표 원자료: `S3_CORR/v2/fifo_casebook.md` (p50 사례 replay 31 L8→9, p95 사례 replay 95 L10→11, tail 사례 replay 246 L9→10 numa0 down 5.2 ms).

## 3. 최대 두 후보
### 후보 A — Cold expert 작업 서비스 시간 (AVX vec_mul 경로의 up_gate/down)
- 변경 파일/함수 후보: `kt-kernel/operators/amx/moe.hpp::do_gate_up_gemm/do_down_gemm` 의 분기 규칙 또는 `operators/amx/la/amx_kernels.hpp` 의 vec_mul(AVX-512, RB) 구현. 변경 변수 1개: (a) `KT_AMX_MIN_ROWS` 임계(예: 3) 로 m≥3 expert 를 AMX 로, 또는 (b) vec_mul 커널의 up/gate 융합·타일링 1건.
- 해결하려는 직접 관측: 서비스 p50 902~912 µs 중 up_gate+down 89 % (numa0 합 기준), 게시 지연 315~321 µs.
- 보존 조건: 수치 동등(smoke 4 요청 동일 텍스트, 필요 시 대표 expert 출력 비교), deferred 8·top-k 8·정밀도·HBM/KV·작업량 불변.
- 알려진 위험/기존 실패: IDE_030 에서 `KT_AMX_MIN_ROWS=3`(+`KT_AMX_MIN_QLEN` 무한) 은 qlen>80 체제만 +8 %, C32/C64 변화 없음. 1~2행 expert 는 AMX 타일 패딩으로 더 느렸음 (교차 3행). 이번 관측(rows p50 2~5) 과 대조 필요.
### 후보 B — 관측 비용 기반 Hot 예산 재배치 (합 5,952 고정)
- 변경 파일: `/models/kt/ide070/layer_budget_5952.json` (per_layer) + hotmap_v2 의 층별 logical→physical 순서 (빈도순 유지). 변경 변수 1개: 층간 슬롯 이동 (예: 늦은 비율 1.00·unique cold ≥14 인 층 10/13/20/21/42/53/55 로 +k, 게시가 이른 층 31/46/32 에서 −k).
- 해결하려는 직접 관측: 층별 게시 지연 p50 (S3: 층 20 784, 42 676, 21 663, 13 622 µs vs 층 31 −76, 46 −2, 32 37).
- 보존 조건: HBM 4.08 GB 여유 내 (슬롯 합 고정), hotmap 정합, 정확성.
- 위험: 관측 비용 ≠ 한계 이득; 한 층의 cold 감소가 다음 층 예산에 미치는 효과는 체인(직렬 FIFO)이라 비선형.

## 4. 다음 검증 기준
- 새 MAIN3 OFF 대조 vs 후보 3회 (같은 부팅·순서 교대), 처리량·TTFT/TPOT p95.
- C1 (16 요청)·LONG (32 요청) 회귀 1회씩.
- 정확성: smoke 4 요청 텍스트 동일, 완료율 100 %, NaN/stale 0, lifecycle 미기록 ≤0.1 %.
- 저간섭 계측 (CORE+CORR 1회): 같은 구간(서비스 span, 게시 지연, 층별 표)이 줄었는지 직접 확인.

## 5. 제외한 후보와 이유
| 후보 | 이유 |
|---|---|
| FIFO/dispatch/신호 | 미분리 gap p50 0.75 µs, dequeue→exec 0 → 근거 없음 |
| 복사 묶음화 | HtoD 32 µs·DtoH 는 hot 이전 완료 → 노출 비중 작음 |
| GPU Hot 커널 | hot 이 늦은 표본 10 % 뿐 |
| NUMA/worker 배치 | tail 10/15.6k, 원인 미분리 (FOCUS 예산 없음) |
| TP 통신 | 위치 귀속 미확정, 조건부 미실행 |
| prefill/chunk | EXTEND 분절 없음 |
| GLM | 정상 출력 차단 (G00 별도 트랙) |
| DRAM 기반 변경 | 부하 창 read 233 GB/s 는 관측값, 포화 비교 기준 없음 |
