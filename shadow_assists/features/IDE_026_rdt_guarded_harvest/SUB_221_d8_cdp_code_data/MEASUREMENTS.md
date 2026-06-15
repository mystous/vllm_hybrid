# SUB_221 — CDP code/data 분리, 2026-06-15

> **판정: 보류 (전용 세션 필요) — 미시험.**

## 미수행 사유
- CDP 활성화는 resctrl `umount` 후 `mount -o cdp` **재마운트** 필수 (CLOS 15→7 반감).
- IDE_026 CLAUDE.md 규칙 #10: "이미 마운트된 resctrl 을 umount 하지 말 것 — 다른
  사용자가 쓸 수 있음" → **공유 호스트에서 실행 불가**. README 도 "별도 세션에서만".
- 후순위·탐색적(D8). 기각 아님(가설 미검증).

## 설계 (전용 세션에서)
- CDP 재마운트 후 {L3CODE 보호, L3DATA 보호, 둘 다} × harvest aggressor → victim p99 비교.
- 가설: detok/CPython 코드 footprint 가 커서, harvest 스트리밍의 코드라인 evict 가
  frontend stall → L3CODE 보호가 data 분할보다 효율적일 수 있음.
- 게이트: CODE-보호 단독이 동등 way 예산 data 분할보다 p99 우수 → 채택.

산출물: (설계만)

---

## [측정 2026-06-15] CDP 재마운트 후 실험 (보류 해제)

`mount -o cdp` 성공(schemata L3CODE/L3DATA 분리 확인). resctrl group `sub221` 에
harvest aggressor 할당, L3CODE/L3DATA 마스크별 victim 간섭 측정.

### 결과 (victim 0-7 + aggressor 8-23 basic, baseline ns/load=85.6)
| 셀 | L3CODE | L3DATA | victim ns/load |
|---|---|---|---|
| unrestricted | full(fffff) | full(fffff) | 207.6 |
| data_restricted | full | 2-way(00003) | 206.0 |
| code_restricted | 2-way(00003) | full | 205.9 |

### 판정
- harvest 의 L3 CODE/DATA way 를 20→2 로 제한해도 **victim 간섭 변화 없음** (전부 ~206).
- **L3 CDP/CAT 캐시 파티셔닝은 BW-bound harvest 에 무효**: aggressor 32MB 스트리밍이
  메모리 BW 포화 → LLC way 할당 축소가 DRAM BW 간섭을 못 줄임. **MBA(BW)가 실효 레버**
  (SUB_220/224 와 정합).
- D8 의 "code footprint 보호(L3CODE)" 가설은 **data-bound 합성 victim 이라 직접 검증 불가**.
  실 CPython 서빙(code-heavy)이면 L3CODE 보호가 다를 수 있으나, BW-bound harvest 하에선
  캐시 파티션 자체가 부차적. → CDP 는 harvest 가드로 비효율(MBA 우선).

### HW 복구
- 측정 후 `sub221` group rmdir. resctrl 은 사용자가 non-cdp 로 재마운트 복구 (CLOS 7→15).
- (주: 이 실험 중 DSA wq0.1 은 device-locked 로 disabled 잔존 — lhc 기각 트랙, 무해.)
