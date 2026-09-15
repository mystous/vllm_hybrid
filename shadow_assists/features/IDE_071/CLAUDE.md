# IDE_071 실행 규칙 (지시서 §0)
- 중간 종료 금지. 셀 실패는 기록 후 다음 독립 셀. BIOS 잠금·권한 부족은 우회하지 않고 BLOCKED_* 로 종료.
- 최종 문서에 평가·해석·권고 금지. 측정값·설정·코드 변경 사실·오류 원문·상태만.
- 셀 순서 고정 seed 무작위화, 6셀/90분 앵커 재측정. 탐색 셀 3회, 최종 후보 5~9회.
- PERF_CLEAN 과 DIAGNOSTIC 분리. profiler 포함 처리량은 일반 평균에 넣지 않음.
- 게시: `git add` 경로 명시, force push 금지, data commit 과 receipt commit 분리.
- 컨테이너 sgl-kt / vllm-h100 (nerdctl 셔임 ~/bin/docker). vllm-h100 은 ~/.cache/huggingface 를 같은 경로로 마운트, /models 없음.
