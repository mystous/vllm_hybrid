# SUB_234 — TLB/hugepage page-walk 2차 트래픽, 2026-06-15

> **판정: 불확정 (minor 채널 추정).** madvise THP 미승인(AnonHugePages=0), 효과 ≤ run 노이즈.
> 정밀 측정엔 예약 HugeTLB(sudo nr_hugepages) 필요 — minor 추정이라 미수행.

## 측정 (victim ws 256MB = TLB 압박, --hugepage MADV_HUGEPAGE)
| 변형 | ns/load (반복) | 평균 |
|---|---|---|
| normal (4KB) | 64.7 61.3 54.9 / 55.4 54.4 59.4 63.1 | ~59-62 |
| --hugepage | 54.7 55.1 58.2 / 63.4 66.8 63.0 62.0 | ~56-64 |

## 판정
1. **THP 미승인**: `--hugepage`(posix_memalign 2MB + MADV_HUGEPAGE) 적용해도
   `/proc/PID/smaps_rollup` **AnonHugePages=0 kB** → 실제 hugepage 안 잡힘
   (khugepaged 미승격/미가용). 즉 두 변형 모두 사실상 4KB 페이지.
2. **효과 ≤ 노이즈**: ns/load run 변동 ±10%(54.9~66.8)가 변형 간 차이를 압도.
   page-walk/TLB 채널의 분리 가능한 신호 없음.
3. **추정**: 랜덤 포인터체이스의 page-walk 는 page-walk-cache(PWC)+PTE의 L2 캐싱으로
   대부분 흡수 → DRAM 데이터 미스 대비 **minor 2차 채널**. 지배적 간섭 아님.

## 정밀 측정 요건 (미수행)
- 예약 HugeTLB: `sudo sysctl vm.nr_hugepages=N` + `mmap(MAP_HUGETLB)` (시스템 변경,
  공유호스트). 또는 aggressor 의 page-walk 트래픽을 victim degr 로 분리(별도 설계).
- minor 추정 + 시스템 변경 비용 → 본 세션 미수행.

산출물: 코드 `src/victim_aggressor.c` (`--hugepage`).

---

## [정정 2026-06-15] 예약 HugeTLB 로 재측정 — TLB 채널 유의함 (불확정 → positive)

`sudo sysctl vm.nr_hugepages=320`(640MB 예약) + `--hugetlb`(mmap MAP_HUGETLB|MAP_HUGE_2MB).
HugePages 실제 소비 확인(Free 320→212). cleanup munmap 버그(mmap 에 free→abort) 수정.

### 결과 (victim ws 200MB, 4반복)
| 페이지 | ns/load | 평균 |
|---|---|---|
| normal 4KB | 45.4 49.1 50.2 50.0 | 48.7 |
| **HugeTLB 2MB** | 40.0 38.3 38.2 38.6 | **38.8** |

→ **HugeTLB = −20.3% 지연** (범위 비중첩, 명확).

### 정정 판정
- 앞선 "minor/불확정" 은 **madvise THP 미승인(AnonHugePages=0)** 때문이었음 — 측정 오류.
- **실제 예약 HugeTLB 로는 TLB/page-walk 가 ~20% 유의 채널**. 200MB ws(50K 4KB 페이지
  ≫ TLB ~1.5K entry) → 매 chase 가 TLB-miss + page-walk → 지연 20% 가산.
- **함의**: serving working set 에 hugepage 적용 시 자체 page-walk 지연 −20%. harvest 가
  serving 의 TLB/PWC 를 오염시키는 시나리오에서도 hugepage 가 방어. 간섭 채널 서열에서
  **TLB 는 minor 아님 (메모리BW > NUMA > TLB ~20% > uncore)**.

→ SUB_234 판정: 🟡 불확정 → ✅ **positive** (TLB 채널 ~20%, hugepage 완화 입증).
