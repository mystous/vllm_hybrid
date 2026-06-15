# LHC Phase 3 — Task B: WQ-per-rank wrapper + TP=8 EBUSY 해결

**날짜**: 2026-06-08
**상위**: `lhc_phase3/PHASE3_VERDICT.md`
**산출물**: `vllm/v1/lhc/dsa_lane.py`, `vllm/v1/lhc/libdsa_lane.c`, `lhc_phase3/test_wq_per_rank{,_concurrent}.py`

---

## 0. TL;DR

8 child proc (TP=8 emulate) 동시 64 KB × 200 iters DSA self-test:
- **8/8 PASS, 0 EBUSY, 0 fails** (각 proc 평균 ≈ 16 GB/s, host-only memcpy)
- rank → WQ mapping (dsa0/wq0.0~0.3 + dsa1/wq1.0~1.3) 정상 동작
- 호스트 enable 한 8 WQ 가 **shared mode (SWQ)** 임이 발견 → `libdsa_lane.c` 에 **ENQCMD path** 추가 (PASID-tagged shared submit)

Phase 2 의 `EBUSY` 단일 wq0.0 공유 문제는 (1) WQ-per-rank mapping + (2) SWQ + ENQCMD 두 변경으로 완전 해소.

---

## 1. 발견 — 호스트 WQ 가 SHARED mode

```
$ cat /sys/bus/dsa/devices/wq0.0/mode
shared
```

8 WQ 전체 (`wq0.0~0.3 + wq1.0~1.3`) mode=shared, type=user. Phase 2 의 라이브러리는 dedicated WQ + `MOVDIR64B` 만 지원했기 때문에 self-test 가 1 sec polling timeout (`rc=-3`) 으로 실패.

SWQ 는 `ENQCMD` (PASID-tagged) 로 submit 해야 함. SWQ 모드에서:
- 8 proc 가 **동일** WQ 를 공유해도 OS PASID 가 descriptor 별 분리 → EBUSY 없음.
- 단, 본 작업은 추가 안전마진을 위해 **rank별 다른 WQ** 도 함께 적용 (rank R → `/dev/dsa/wq{R//4}.{R%4}`).

---

## 2. lib 변경 (`libdsa_lane.c`)

추가 사항:
1. **auto-detect SWQ vs DWQ** — `/sys/bus/dsa/devices/<wq>/mode` 읽어 `g_is_shared` 셋업.
2. **ENQCMD inline asm** (`.byte 0xf2,0x0f,0x38,0xf8,0x02`, opcode = enqcmd r/m64 → rax).
   - ZF=1 → SWQ full → bounded retry (max 100 K spin), `_mm_pause()` between.
   - 실패 시 `g_fail++`, return `-4`.
3. **`arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_PASID)`** — XSAVE-PASID component 요청. 신규 커널 호환. 본 머신 (kernel 6.8) 은 `ENOSYS` 반환하나 ENQCMD 는 정상 동작 (PASID 자동 발급).

dedicated path (`MOVDIR64B`) 는 유지 — DWQ 환경에서도 그대로 동작.

빌드:
```
gcc -O3 -march=native -mmovdir64b -fPIC -shared -o libdsa_lane.so libdsa_lane.c -pthread
```

(`-menqcmd` 는 inline asm 사용으로 불필요)

---

## 3. wrapper 변경 (`dsa_lane.py`)

(Phase 2 이미 작성된 `_resolve_dev_path()` 그대로 사용)

env:
- `VLLM_LHC_DSA_WQ_PER_RANK=1` — rank → WQ 매핑 활성
- `VLLM_LHC_DSA_RANK` — 명시 rank (otherwise `LOCAL_RANK`/`RANK`/0)

매핑:
| rank | dev | NUMA |
|---|---|---|
| 0 | `/dev/dsa/wq0.0` | node 0 |
| 1 | `/dev/dsa/wq0.1` | node 0 |
| 2 | `/dev/dsa/wq0.2` | node 0 |
| 3 | `/dev/dsa/wq0.3` | node 0 |
| 4 | `/dev/dsa/wq1.0` | node 1 |
| 5 | `/dev/dsa/wq1.1` | node 1 |
| 6 | `/dev/dsa/wq1.2` | node 1 |
| 7 | `/dev/dsa/wq1.3` | node 1 |

---

## 4. 검증 결과

### 4.1 unit test (`test_wq_per_rank.py`)
```
[T1] default → wq0.0  OK
[T2] explicit dev override  OK
[T3] rank=0..7 → wq{0..1}.{0..3}  OK
[T4] LOCAL_RANK=5 → wq1.1  OK
[T5] disabled gate fallthrough  OK
ALL PASS
```

### 4.2 8-proc concurrent (`test_wq_per_rank_concurrent.py`)
- 8 spawn proc, 각 64 KB × 200 iters MEMMOVE
- payload 13.1 MB / proc, total 105 MB

```
rank dev                       avail   ok    ops  fails    BW_GBps
   0 /dev/dsa/wq0.0             True True    201      0     17.618
   1 /dev/dsa/wq0.1             True True    201      0     16.229
   2 /dev/dsa/wq0.2             True True    201      0     16.410
   3 /dev/dsa/wq0.3             True True    201      0     16.282
   4 /dev/dsa/wq1.0             True True    201      0     14.468
   5 /dev/dsa/wq1.1             True True    201      0     16.451
   6 /dev/dsa/wq1.2             True True    201      0     16.726
   7 /dev/dsa/wq1.3             True True    201      0     17.926
results: 8/8  fails: 0
```

per-proc BW ≈ 16 GB/s, 8 proc 합산 ≈ 130 GB/s (host DRAM memcpy 합). NUMA node 1 (rank 4) 만 14.4 GB/s 로 약간 낮음 — child spawn 시 NUMA-local affinity 미적용 — Task E 에서 `numactl --cpunodebind` 로 추가 향상 측정 예정.

`ops=201` (200 iters + 1 self-test init) — 모든 submit 성공.

---

## 5. Task B verdict

| 항목 | 결과 |
|---|---|
| WQ-per-rank mapping 정확성 | PASS (8/8 expected) |
| 8 proc 동시 self-test no-EBUSY | PASS (0 fails) |
| ENQCMD on SWQ functional | PASS (modus operandi 확립) |
| Phase 2 EBUSY 재현 | 해소 |

→ **Task B PASS**. Task E (multi-engine integrated BW 측정) 진입.

---

## 6. 산출물 위치
```
vllm/v1/lhc/dsa_lane.py             ← _resolve_dev_path() (Phase 2 작성, Phase 3 그대로 사용)
vllm/v1/lhc/libdsa_lane.c           ← SWQ + ENQCMD path 추가 (Phase 3 신규)
vllm/v1/lhc/libdsa_lane.so          ← rebuilt
lhc_phase3/test_wq_per_rank.py       ← unit test
lhc_phase3/test_wq_per_rank_concurrent.py  ← 8-proc concurrent test
lhc_phase3/task_b_wq_per_rank.md     ← 본 문서
```
