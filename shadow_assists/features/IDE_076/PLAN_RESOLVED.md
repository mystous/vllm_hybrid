# PLAN_RESOLVED — IDE_076 (PLAN.md 를 실행 소스에서 확정한 값; 2026-09-17 23:05 KST, O0 시점. 단계 진행에 따라 갱신)

| PLAN.md 미확정 항목 | 확정 값 / 근거 |
|---|---|
| 실행 소스·현재 HEAD | 저장소 edfb8bb99 (브랜치 feat/cpu-moe-a1a2b-20260917). kt-kernel: upstream 6d460cc + 로컬 11 파일 (SOURCE_MANIFEST.json). SGLang 71de97b + 로컬 10 파일 |
| 정상성 수정본 바이너리 | v4 `12926df2c30b724694171108635637e8cda14064ae9a643c03b114d6c9d6b926` (기록기 v3 + FP8/BF16 dispatch 수정). reference_binary |
| 실제 kernel shape | NUMA 서브풀 2: gate/up K=6,144·N=1,280 (nth 10), down K=1,280·N=6,144 (nth 48); INT4 tile 32×32×64, K_BLOCK 3,584, B tile 2 KB |
| bs=63 Cold job 경로 | qlen=64 → forward_prefill → vec_mul → RB(avx_kernel_rb) (AMX 임계 KT_AMX_MIN_QLEN 80 기본, KT_AMX_MIN_ROWS 1<<30) |
| 기존 RB 의 R=1/2 처리 | 정적 블록 `avx_rb_rows<1,4>` / `<2,2>` 존재 (ALREADY_IMPLEMENTED: row blocking 자체). 잔여 비용은 B 언팩(k_i 당 6 op)·A 브로드캐스트·gate/up 이중 입력 접근·to_mat 별도 pass (A1_GAP_ANALYSIS §3~4) |
| A2 대상 구조 | job 당 서브풀별 `do_work_stealing_job` 7회 (S0 gather, S1 A 양자화, S2 gate/up 20 task/expert, S3 act, S4 down 양자화, S5 down 48 task/expert, S6 가중합), 장벽 = 워커별 mutex+cv notify + 부모 spin wait, guided block=1 (CODE_MAP §1). S29 스테이지: GEMM 2개 88 %, 나머지 ≈94 µs |
| 횟수 guard | IDE_075 하네스의 BOOT/SESSION 상한 분기를 EXT=True 로 비활성 (EXECUTION_POLICY_TESTS.md); retry/quality/시간 상한 없음 |
| 새 무계측 기준선 | R0_REF (v4·H0): MAIN 745.4/746.0/747.5, C1 53.2, LONG 163.0 tok/s (WORK_LOG 22:49). 신규 빌드 R0 는 플래그 OFF 로 재측정 후 대조 |
| 채택 기준 | ACCEPTANCE_CRITERIA.yaml (최소 이득 2 %, 블록 짝 비교, p95 회귀 한도, 정수 계약 bitwise) |
| 워크로드 manifest | eval/ide071/configs/workloads.json + ~/.cache/huggingface/kt/ide073/inputs/qwen/*.jsonl (SHA 는 manifest); token 합계 R0_REF 에서 표와 일치 (131,786/32,768 · 8,234/2,048 · 131,147/4,096) |
| 클라이언트 | vllm bench serve (vllm-h100 0.26.1rc1) / vllmw 래퍼 (SHA 16046514…, 동등 확인 IDE_075) |
| 계측 모드 대응 | OFF / CORR(KT_EVT + torch profiler) / RESOURCE / FOCUS = IDE_075 하네스 MODE (기능 플래그와 독립) |
| 신규 플래그 계약 | `KT_OPT_A1_ENABLE`, `KT_OPT_A2_ENABLE` (0/1 명시, 미지정 0, 잘못된 값 → 초기화 실패), policy 파일 해시 기록 — 구현 C03 |
| 데이터 집합 분리(B) | CALIBRATION = PROBE128 CORR 계측(expert·job 비용), SELECTION = 별도 PROBE128 seed 변형 + MAIN 블록, CONFIRMATION = 동결 후 새 부팅 MAIN/C1/LONG. 상세는 B 단계 시작 시 CALIBRATION_MANIFEST.json |
| 이번에 하지 않을 일 | PLAN.md §0 목록 그대로 (GPU 수 변경, 전 CPU 배치, AMX threshold 전수 탐색, 워커/NUMA 전수 탐색, 전역 FIFO 교체 등) |
