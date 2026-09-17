# IDE_075 — Claude 작업 메모

- IDE_074 CLAUDE.md 의 노드·컨테이너·규칙을 그대로 적용. 추가:
- 하네스 v2(`eval/ide075/harness.py`)는 IDE_074 하네스를 `importlib` 로 별도 이름(H74)으로 로드한다. `import harness` 로 쓰면 자기 자신을 가리켜 부팅을 낭비한다 (09-17 15:27 사고).
- 체인 스크립트에서 다음 plan 은 `stop_server()` 로 이전 서버를 죽인다. 하네스가 죽었더라도 서버가 살아 있으면 `ATTACH:<plan>:<boot_id>` 로 재사용 (새 부팅 아님).
- 예산 계수는 boot_start(시도) 기준. 잔여 예산은 IDE_074 원장과 합산 (`usage()`).
- kt_evt v2 파일: `kt_evt.csv`(패킷 레코드), `.tasks`(task 링), `.map`(slot→layer, effective_from=ns). 서버 종료 시 footer `#dropped=…`. 세션 슬라이스는 t_go 창.
- 분석 순서: `dependency_v2.py <sess> --v2` → `analyze_costs_v2.py` → (`resource_v2.py` RESOURCE) → `validate_v2.py` → `render_report_v2.py`. READINESS/OPTIMIZATION_HANDOFF 는 수동.
- pcm-memory csv 는 로컬 시각(Asia/Seoul)·소수초; `tsparse.local_timestamp_ns` 로만 파싱. 표본 구간은 [ts−1 s, ts) 가정(ASSUMED_END_TIMESTAMP).
