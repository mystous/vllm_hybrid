
## 보고 0 — 예정 2026-09-17T07:45:00+0900 / 실제 2026-09-17T07:55:33+0900 (경과 633 s)
- 단계/현재: PREPARING / GLM-4.7-FP8 다운로드 + 환경 동결 + 하네스 준비
- 항목 수: {'registered': 0}
- 최근 30분 완료: None
- GLM 다운로드 bytes: 43963106757
- HBM MiB: 0, 0 1, 0 2, 0 3, 0 4, 0 5, 0 6, 0 7, 0
- top RSS: hf:4244MB kube-apiserver:2550MB · 디스크 여유: 21T
- 오류: None
- 다음: None · 잔여 추정: None
- 전달: saved (세션 cron 이 전달)
- 08:12 Q-GPU8 a1: 벤치 클라이언트 pandas 미설치로 custom dataset 로드 실패(요청 0건) → 세션 void 재분류, vllm-h100 에 pandas 설치(클라이언트 측 변경), Q-KT-BASIC4 a1 부팅 중단(부팅 1 소비). 재시작.

## 보고 1 — 예정 2026-09-17T08:15:00+0900 / 실제 2026-09-17T08:15:15+0900 (경과 1815 s)
- 단계/현재: qwen/GPU8 / {'cell': 'Q-GPU8', 'model': 'qwen', 'profile': 'GPU8', 'attempt': 'a2', 't': '2026-09-17T08:12:40.866279+09:00'}
- 항목 수: {'completed': 0, 'failed': 1, 'blocked': 0, 'registered': 1}
- 최근 30분 완료: ["Q-GPU8:FAILED_REQUESTS reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 0), ('M073_QWEN_MAIN_SHORT_C64_rep2', 0), ('M073_QWEN_MAIN_SHORT_C64_rep3', 0), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 0), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 0)]"]
- GLM 다운로드 bytes: 177181242161
- HBM MiB: 0, 63752 1, 63848 2, 63848 3, 63848 4, 63848 5, 63848 6, 63848 7, 63368
- top RSS: hf:5832MB sglang::schedul:3264MB · 디스크 여유: 21T
- 오류: ['Q-GPU8: rc=1 completed=None failed=None n=16', 'Q-GPU8: rc=1 completed=None failed=None n=32']
- 다음: Q-GPU8 자격·성능 → Q-KT-BASIC4 → Q-OPT4 → Qwen 품질·프로브 → (GLM 다운로드 완료 후) G-GPU8 → G-KT-BASIC4 → G 변환/OPT4 · 잔여 추정: GLM 다운로드 ~75 min (80 MB/s 기준) · Qwen 3구성 ~2.5 h · GLM 3구성+변환 ~3.5 h → 전체 종료 추정 9-17 15~16시 (준비·부팅 실측 후 갱신)
- 전달: saved (세션 cron 이 전달)

## 보고 2 — 예정 2026-09-17T08:45:00+0900 / 실제 2026-09-17T08:45:17+0900 (경과 3617 s)
- 단계/현재: qwen/BASIC4 / {'cell': 'Q-KT-BASIC4', 'model': 'qwen', 'profile': 'BASIC4', 'attempt': 'a4', 't': '2026-09-17T08:32:12.513187+09:00'}
- 항목 수: {'completed': 3, 'failed': 1, 'blocked': 0, 'registered': 4}
- 최근 30분 완료: ["Q-GPU8:FAILED_REQUESTS reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 0), ('M073_QWEN_MAIN_SHORT_C64_rep2', 0), ('M073_QWEN_MAIN_SHORT_C64_rep3', 0), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 0), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 0)]", "Q-GPU8:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 1555.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 1553.6), ('M073_QWEN_MAIN_SHORT_C64_rep3', 1555.7), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 82.5), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 267.4)]", 'Q-KT-BASIC4:FAILED_BOOT reps=[]', 'Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 63.1), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 162.3)]", "Q-OPT4-b2:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 705.7), ('M073_QWEN_MAIN_SHORT_C64_rep2', 749.7)]"]
- GLM 다운로드 bytes: 362091875343
- HBM MiB: 0, 79432 1, 79342 2, 79342 3, 78862 4, 0 5, 0 6, 0 7, 0
- top RSS: sglang::schedul:458329MB sglang::schedul:3139MB · 디스크 여유: 21T
- 오류: ['Q-KT-BASIC4: DIED']
- 다음: Q-GPU8 자격·성능 → Q-KT-BASIC4 → Q-OPT4 → Qwen 품질·프로브 → (GLM 다운로드 완료 후) G-GPU8 → G-KT-BASIC4 → G 변환/OPT4 · 잔여 추정: GLM 다운로드 ~75 min (80 MB/s 기준) · Qwen 3구성 ~2.5 h · GLM 3구성+변환 ~3.5 h → 전체 종료 추정 9-17 15~16시 (준비·부팅 실측 후 갱신)
- 전달: saved (세션 cron 이 전달)

## 보고 3 — 예정 2026-09-17T09:15:00+0900 / 실제 2026-09-17T09:15:19+0900 (경과 5419 s)
- 단계/현재: glm/GPU8 / {'cell': 'G-GPU8', 'model': 'glm', 'profile': 'GPU8', 'attempt': 'a1', 't': '2026-09-17T09:06:23.689382+09:00'}
- 항목 수: {'completed': 4, 'failed': 0, 'blocked': 0, 'registered': 4}
- 최근 30분 완료: ["Q-GPU8:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 1555.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 1553.6), ('M073_QWEN_MAIN_SHORT_C64_rep3', 1555.7), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 82.5), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 267.4)]", 'Q-KT-BASIC4:FAILED_BOOT reps=[]', 'Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 63.1), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 162.3)]", "Q-OPT4-b2:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 705.7), ('M073_QWEN_MAIN_SHORT_C64_rep2', 749.7)]", "Q-KT-BASIC4:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 137.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 141.3), ('M073_QWEN_MAIN_SHORT_C64_rep3', 142.7), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 30.4), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 32.8)]"]
- GLM 다운로드 bytes: 362091875343
- HBM MiB: 0, 50070 1, 50166 2, 50166 3, 50166 4, 50166 5, 50166 6, 50166 7, 49686
- top RSS: sglang::schedul:3583MB sglang::schedul:3575MB · 디스크 여유: 21T
- 오류: ['Q-KT-BASIC4: DIED']
- 다음: Q-GPU8 자격·성능 → Q-KT-BASIC4 → Q-OPT4 → Qwen 품질·프로브 → (GLM 다운로드 완료 후) G-GPU8 → G-KT-BASIC4 → G 변환/OPT4 · 잔여 추정: GLM 다운로드 ~75 min (80 MB/s 기준) · Qwen 3구성 ~2.5 h · GLM 3구성+변환 ~3.5 h → 전체 종료 추정 9-17 15~16시 (준비·부팅 실측 후 갱신)
- 전달: saved (세션 cron 이 전달)

## 보고 4 — 예정 2026-09-17T09:45:00+0900 / 실제 2026-09-17T09:45:02+0900 (경과 7202 s)
- 단계/현재: glm/BASIC4 / {'cell': 'G-KT-BASIC4', 'model': 'glm', 'profile': 'BASIC4', 'attempt': 'a1', 't': '2026-09-17T09:16:46.146650+09:00'}
- 항목 수: {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 5}
- 최근 30분 완료: ['Q-KT-BASIC4:FAILED_BOOT reps=[]', 'Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 63.1), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 162.3)]", "Q-OPT4-b2:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 705.7), ('M073_QWEN_MAIN_SHORT_C64_rep2', 749.7)]", "Q-KT-BASIC4:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 137.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 141.3), ('M073_QWEN_MAIN_SHORT_C64_rep3', 142.7), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 30.4), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 32.8)]", "G-GPU8:COMPLETED reps=[('M073_GLM_MAIN_SHORT_C64_rep1', 1238.0), ('M073_GLM_MAIN_SHORT_C64_rep2', 1238.7), ('M073_GLM_MAIN_SHORT_C64_rep3', 1231.5), ('M073_GLM_LOW_CONCURRENCY_C1_rep1', 79.4), ('M073_GLM_LONGER_PREFILL_C8_rep1', 228.8)]"]
- GLM 다운로드 bytes: 362091875343
- HBM MiB: 0, 54898 1, 54444 2, 54444 3, 53964 4, 0 5, 0 6, 0 7, 0
- top RSS: sglang::schedul:328948MB sglang::schedul:3518MB · 디스크 여유: 21T
- 오류: ['Q-KT-BASIC4: DIED']
- 다음: Q-GPU8 자격·성능 → Q-KT-BASIC4 → Q-OPT4 → Qwen 품질·프로브 → (GLM 다운로드 완료 후) G-GPU8 → G-KT-BASIC4 → G 변환/OPT4 · 잔여 추정: GLM 다운로드 ~75 min (80 MB/s 기준) · Qwen 3구성 ~2.5 h · GLM 3구성+변환 ~3.5 h → 전체 종료 추정 9-17 15~16시 (준비·부팅 실측 후 갱신)
- 전달: saved (세션 cron 이 전달)

## 보고 5 — 예정 2026-09-17T10:15:00+0900 / 실제 2026-09-17T10:15:04+0900 (경과 9004 s)
- 단계/현재: glm/BASIC4 / {'cell': 'G-KT-BASIC4', 'model': 'glm', 'profile': 'BASIC4', 'attempt': 'a1', 't': '2026-09-17T09:16:46.146650+09:00'}
- 항목 수: {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 5}
- 최근 30분 완료: ['Q-KT-BASIC4:FAILED_BOOT reps=[]', 'Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 63.1), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 162.3)]", "Q-OPT4-b2:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 705.7), ('M073_QWEN_MAIN_SHORT_C64_rep2', 749.7)]", "Q-KT-BASIC4:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 137.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 141.3), ('M073_QWEN_MAIN_SHORT_C64_rep3', 142.7), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 30.4), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 32.8)]", "G-GPU8:COMPLETED reps=[('M073_GLM_MAIN_SHORT_C64_rep1', 1238.0), ('M073_GLM_MAIN_SHORT_C64_rep2', 1238.7), ('M073_GLM_MAIN_SHORT_C64_rep3', 1231.5), ('M073_GLM_LOW_CONCURRENCY_C1_rep1', 79.4), ('M073_GLM_LONGER_PREFILL_C8_rep1', 228.8)]"]
- GLM 다운로드 bytes: 362091875343
- HBM MiB: 0, 54836 1, 54678 2, 54678 3, 54198 4, 0 5, 0 6, 0 7, 0
- top RSS: sglang::schedul:329006MB sglang::schedul:3518MB · 디스크 여유: 21T
- 오류: ['Q-KT-BASIC4: DIED']
- 다음: Q-GPU8 자격·성능 → Q-KT-BASIC4 → Q-OPT4 → Qwen 품질·프로브 → (GLM 다운로드 완료 후) G-GPU8 → G-KT-BASIC4 → G 변환/OPT4 · 잔여 추정: GLM 다운로드 ~75 min (80 MB/s 기준) · Qwen 3구성 ~2.5 h · GLM 3구성+변환 ~3.5 h → 전체 종료 추정 9-17 15~16시 (준비·부팅 실측 후 갱신)
- 전달: saved (세션 cron 이 전달)

## 보고 6 — 예정 2026-09-17T10:45:00+0900 / 실제 2026-09-17T10:45:06+0900 (경과 10806 s)
- 단계/현재: glm/BASIC4 / None
- 항목 수: {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 6}
- 최근 30분 완료: ['Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 63.1), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 162.3)]", "Q-OPT4-b2:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 705.7), ('M073_QWEN_MAIN_SHORT_C64_rep2', 749.7)]", "Q-KT-BASIC4:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 137.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 141.3), ('M073_QWEN_MAIN_SHORT_C64_rep3', 142.7), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 30.4), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 32.8)]", "G-GPU8:COMPLETED reps=[('M073_GLM_MAIN_SHORT_C64_rep1', 1238.0), ('M073_GLM_MAIN_SHORT_C64_rep2', 1238.7), ('M073_GLM_MAIN_SHORT_C64_rep3', 1231.5), ('M073_GLM_LOW_CONCURRENCY_C1_rep1', 79.4), ('M073_GLM_LONGER_PREFILL_C8_rep1', 228.8)]", "G-KT-BASIC4:None reps=[('M073_GLM_MAIN_SHORT_C64_rep1', 46.4), ('M073_GLM_MAIN_SHORT_C64_rep2', 46.5), ('M073_GLM_MAIN_SHORT_C64_rep3', 46.4), ('M073_GLM_LOW_CONCURRENCY_C1_rep1', 17.5), ('M073_GLM_LONGER_PREFILL_C8_rep1', 7.4)]"]
- GLM 다운로드 bytes: 362091875343
- HBM MiB: 0, 0 1, 0 2, 0 3, 0 4, 0 5, 0 6, 0 7, 0
- top RSS: kube-apiserver:2509MB java:1193MB · 디스크 여유: 21T
- 오류: ['Q-KT-BASIC4: DIED']
- 다음: Q-GPU8 자격·성능 → Q-KT-BASIC4 → Q-OPT4 → Qwen 품질·프로브 → (GLM 다운로드 완료 후) G-GPU8 → G-KT-BASIC4 → G 변환/OPT4 · 잔여 추정: GLM 다운로드 ~75 min (80 MB/s 기준) · Qwen 3구성 ~2.5 h · GLM 3구성+변환 ~3.5 h → 전체 종료 추정 9-17 15~16시 (준비·부팅 실측 후 갱신)
- 전달: saved (세션 cron 이 전달)

## 보고 7 — 예정 2026-09-17T11:15:00+0900 / 실제 2026-09-17T11:07:28+0900 (경과 12148 s)
- 단계/현재: glm/BASIC4 / None
- 항목 수: {'completed': 5, 'failed': 0, 'blocked': 0, 'registered': 6}
- 최근 30분 완료: ['Q-KT-BASIC4:FAILED_BOOT reps=[]', "Q-OPT4-b1:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 689.4), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 63.1), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 162.3)]", "Q-OPT4-b2:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 705.7), ('M073_QWEN_MAIN_SHORT_C64_rep2', 749.7)]", "Q-KT-BASIC4:COMPLETED reps=[('M073_QWEN_MAIN_SHORT_C64_rep1', 137.4), ('M073_QWEN_MAIN_SHORT_C64_rep2', 141.3), ('M073_QWEN_MAIN_SHORT_C64_rep3', 142.7), ('M073_QWEN_LOW_CONCURRENCY_C1_rep1', 30.4), ('M073_QWEN_LONGER_PREFILL_C8_rep1', 32.8)]", "G-GPU8:COMPLETED reps=[('M073_GLM_MAIN_SHORT_C64_rep1', 1238.0), ('M073_GLM_MAIN_SHORT_C64_rep2', 1238.7), ('M073_GLM_MAIN_SHORT_C64_rep3', 1231.5), ('M073_GLM_LOW_CONCURRENCY_C1_rep1', 79.4), ('M073_GLM_LONGER_PREFILL_C8_rep1', 228.8)]", "G-KT-BASIC4:None reps=[('M073_GLM_MAIN_SHORT_C64_rep1', 46.4), ('M073_GLM_MAIN_SHORT_C64_rep2', 46.5), ('M073_GLM_MAIN_SHORT_C64_rep3', 46.4), ('M073_GLM_LOW_CONCURRENCY_C1_rep1', 17.5), ('M073_GLM_LONGER_PREFILL_C8_rep1', 7.4)]"]
- GLM 다운로드 bytes: 362091875343
- HBM MiB: 0, 0 1, 0 2, 0 3, 0 4, 0 5, 0 6, 0 7, 0
- top RSS: kube-apiserver:2552MB java:1193MB · 디스크 여유: 21T
- 오류: ['Q-KT-BASIC4: DIED']
- 다음: Q-GPU8 자격·성능 → Q-KT-BASIC4 → Q-OPT4 → Qwen 품질·프로브 → (GLM 다운로드 완료 후) G-GPU8 → G-KT-BASIC4 → G 변환/OPT4 · 잔여 추정: GLM 다운로드 ~75 min (80 MB/s 기준) · Qwen 3구성 ~2.5 h · GLM 3구성+변환 ~3.5 h → 전체 종료 추정 9-17 15~16시 (준비·부팅 실측 후 갱신)
- 전달: saved (세션 cron 이 전달)

## 11:10 종료 보고
- 상태 COMPLETED_WITH_FAILURES · 게시 PUBLISHED (data fb3675577 / receipt 67d24dd84) · 파일 5개 전달 · 30분 보고 cron 삭제
