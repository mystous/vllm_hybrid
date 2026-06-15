# LHC Phase 2 — Task 1: DSA multi-engine BW gate (측정)

**날짜**: 2026-06-08
**대상**: `/dev/dsa/wq0.0` (컨테이너 노출 single WQ), dsa0 group0
**목표**: aggregate BW ≥ 0.8 × cudaMemcpyAsync (≥ 44 GB/s, baseline 55 GB/s)

---

## 0. TL;DR

| metric | 측정값 | gate (≥0.8× = 44 GB/s) | verdict |
|--------|--------|------------------------|---------|
| descriptor pipelining (depth 1→32, single thread) | **31.37 GB/s plateau** | FAIL (0.57×) |
| multi-thread concurrent submit (1→16 thread, same WQ) | **31.37 GB/s plateau** | FAIL (0.57×) |
| **HW 한계 진단** | single engine cap (group0.0 = engine0.0 단독) | — |

→ **single WQ = single engine = 31.37 GB/s 가 hard ceiling**.
→ multi-engine binding (group0 에 engine0.1~0.3 추가, 또는 dsa1 enable) 은 호스트 root + sysfs RW 필요. 컨테이너에서는 sysfs 가 read-only → 본 phase 에서 수행 불가.

**결론**: gate **CONDITIONAL FAIL** — HW 능력은 PoC 단계에서 보강 (Task 3 에서 single-engine 31 GB/s + CPU stall 0% 의 ortho-lane 가치만으로도 진행). 호스트 enable 후 multi-engine retry 는 별도 1-shot 작업.

---

## 1. 측정 결과

### 1.1 Descriptor pipelining sweep — `dsa_multi_engine_bench.c`

| total bytes | depth=1 | depth=2 | depth=4 | depth=8 | depth=16 | depth=32 |
|------------:|--------:|--------:|--------:|--------:|---------:|---------:|
|   1 MB | 30.21 GB/s | 30.33 | 30.26 | 30.23 | 30.30 | — |
|  16 MB | 31.39 | 31.38 | 31.38 | 31.38 | 31.37 | 31.37 |
|  64 MB | 31.36 | 31.35 | 31.35 | 31.36 | 31.36 | 31.36 |
| 256 MB | 31.37 | 31.37 | 31.37 | 31.37 | 31.37 | 31.37 |

→ depth 늘려도 ±0.02 GB/s 변동 → **단일 dispatch path 가 single engine 으로만 흐름**.

### 1.2 Multi-thread concurrent submit — `dsa_mt_bench.c`

| chunk | thr=1 | thr=2 | thr=4 | thr=8 | thr=16 |
|------:|------:|------:|------:|------:|-------:|
|  1 MB | 29.73 GB/s | 31.02 | 31.12 | 31.33 | 31.35 |
| 16 MB | 31.16 | 31.29 | 31.32 | 31.34 | 31.37 |

→ thread 수에 무관하게 **동일 ceiling**. submit-side contention 아니고 engine HW 한계.

### 1.3 sysfs 상태 (원인 진단)

```
group0.0:   engine0.0  wq0.0      ← 단 1 engine bind
group0.1:   (empty)
group0.2:   (empty)
group0.3:   (empty)
engine0.1/group_id = -1            ← unassigned
engine0.2/group_id = -1
engine0.3/group_id = -1
dsa1/state = disabled              ← 두번째 device 전체 off
wq0.1~wq0.7 = disabled
wq1.0~wq1.7 = disabled
```

→ wq0.0 단독에 engine0.0 만 묶여 있어 BW 가 single-engine 31.4 GB/s 로 고정. 4 engine 묶으면 이론적 ~125 GB/s (PCIe Gen5 x8 도달 가능 영역).

---

## 2. 호스트 enable 절차 (1회 root 작업)

컨테이너 외부 호스트에서 (sudo 권한 필요):

```bash
# dsa0: 4 engine 모두 group0 에 묶고 4 WQ enable
sudo accel-config disable-device dsa0
sudo accel-config config-wq dsa0/wq0.0 --group-id=0 --mode=dedicated --type=user \
     --name=lhc0 --priority=10 --block-on-fault=1 --wq-size=64 \
     --max-transfer-size=2147483648 --max-batch-size=1024
for w in 1 2 3; do
  sudo accel-config config-wq dsa0/wq0.$w --group-id=0 --mode=dedicated --type=user \
       --name=lhc$w --priority=10 --block-on-fault=1 --wq-size=64 \
       --max-transfer-size=2147483648 --max-batch-size=1024
done
for e in 0 1 2 3; do
  sudo accel-config config-engine dsa0/engine0.$e --group-id=0
done
sudo accel-config enable-device dsa0
for w in 0 1 2 3; do sudo accel-config enable-wq dsa0/wq0.$w; done

# dsa1: NUMA node1 측, 동일 절차
sudo accel-config disable-device dsa1
for w in 0 1 2 3; do
  sudo accel-config config-wq dsa1/wq1.$w --group-id=0 --mode=dedicated --type=user \
       --name=lhc1$w --priority=10 --block-on-fault=1 --wq-size=64 \
       --max-transfer-size=2147483648 --max-batch-size=1024
done
for e in 0 1 2 3; do
  sudo accel-config config-engine dsa1/engine1.$e --group-id=0
done
sudo accel-config enable-device dsa1
for w in 0 1 2 3; do sudo accel-config enable-wq dsa1/wq1.$w; done

# 컨테이너 docker run 옵션 보강 (재기동 시)
#   --device=/dev/dsa --cap-add=SYS_RAWIO --cap-add=IPC_LOCK
```

이후 컨테이너에서 `/dev/dsa/wq0.{0..3}` + `/dev/dsa/wq1.{0..3}` 노출 → 8 WQ × 8 engine aggregate ~250 GB/s 가능 (예측).

---

## 3. PoC 진행 결정

| 옵션 | 판단 |
|------|------|
| (A) 호스트 enable 대기 후 Task 1 재측정 → Task 3 진입 | 사용자 root 작업 1건 필요. 별도 트랙 |
| (B) **현재 single engine (31.37 GB/s, CPU stall 0%) 로 Task 3 PoC 진입** | 채택. Phase 1 verdict 의 "조건부 GO — CPU stall 게이트 압도적 PASS" 와 정합. KV swap PoC 의 net-win 은 BW 절대값보다 **CPU 코어 free 효과** 가 dominant |
| (C) DSA 폐기 | 기각 — single engine 도 cudaMemcpy 의 ortho-lane 가치 (PCIe ↔ HBM 사용 안 함, GPU 부담 0) 유지 |

→ **결정: (B)**. Phase 2 PoC 는 single engine 으로 진행하되 PHASE2_VERDICT 에 multi-engine retry 항목 명시.

---

## 4. 산출물

```
/workspace/host_vllm_hybrid/lhc_phase2/
├── dsa_multi_engine_bench.c   ← descriptor pipelining sweep
├── dsa_multi_engine_bench     ← built ELF
├── dsa_multi_engine_raw.jsonl ← raw measurements
├── dsa_mt_bench.c             ← multi-thread submit sweep
├── dsa_mt_bench               ← built ELF
└── dsa_mt_raw.jsonl           ← raw measurements
```
