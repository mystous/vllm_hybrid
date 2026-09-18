# EXECUTION_POLICY_TESTS — IDE_076 (PLAN.md §3.5, 부록 B.1 "실행 정책")

## 조사 결과 (현재 runner/planner/reporter 의 횟수 guard)
| 위치 | guard | 처리 |
|---|---|---|
| `eval/ide075/harness.py` boot() L104 | `BOOT_REMAINING <= 0 and not EXT → SystemExit` | IDE_076 러너는 `H.EXT=True` 를 명시 설정 → 분기 비활성 (숫자 확대 아님) |
| `eval/ide075/harness.py` session() L130 | `SESSIONS_REMAINING <= 0 and not EXT → SystemExit` | 동일 |
| `usage()` 의 IDE_074 원장 합산 (L74) | 합성 상한 계산에만 사용 | 러너에서 `H.L74=/nonexistent` (과거 원장 수정 없음, 참조 안 함) |
| render/report 의 예산 표 | 보고용 계수 | IDE_076 보고서는 `unresolved_required_questions` 기준 (§16.5) |
| retry cap / quality cap / 시간 예산 | harness 에 없음 (bench timeout 1,200 s 는 무진행 복구용, 유지) | 유지 |

안전 검사 유지: 부팅 HEALTH_OK 판정, `server_died` 감지 시 plan 중단, drain 확인, 자기 소유 서버만 stop, `pkill` 전역 사용 안 함.

## 합성 원장 시험
`execution_policy_test_results.json`: 세션 30·부팅 15 (옛 상한 24·12 초과) 원장에서 `boot_guard_would_block=false`, `session_guard_would_block=false`. 신규 0/1 플래그 파서: 미지정→0, "0"/"1" 명시 파싱, "true" → ValueError (부팅 전 실패).

새 원장: `eval/results/IDE_076_20260917/state/execution_events.jsonl` (event 에 `policy: evidence_driven_no_count_cap` 기록). 계획 해시: PLAN.md SHA256 bc2a126aab38e7c99b89b7fa78232e515b0558e1258e6ced7d2d944606ff3d96.
