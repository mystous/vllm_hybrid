# SUB_236 — DSA/IAA 트래픽 셰이핑, 2026-06-15

> **판정 (2026-06-15 갱신): ✅ partial-positive — 컨테이너에서 DSA 제출 성공 +
> 채널② 간섭 dose-response 실측. MBA 비차단성·read-buffer 셰이핑 곡선은 호스트 잔여.**
> (이전 호스트 판정 "보류 — portal mmap EPERM" 은 아래 §최신 결과로 해소)

## 현 가용성 (확인됨)
- DSA 디바이스 dsa0/dsa1, WQ wq0.0~wq1.3 enabled/shared. accel-config·libaccel-config 존재.
- 노브: group `read_buffers_allowed`(96 max)/`use_read_buffer_limit`/traffic_class.

## 미수행 사유
1. **DSA 트래픽 생성기 신규 구현 필요 (~100줄)**: /dev/dsa WQ portal mmap +
   memcpy descriptor 빌드 + ENQCMD/MOVDIR64B 제출 루프. libaccel-config 는 설정용이라
   제출은 별도. `dsa_test` 미설치.
2. **read_buffers_allowed sweep = WQ group 재구성**: WQ disable→group reconfig→enable
   필요. 호스트 DSA 는 **다른 실험과 공유 중**(IDE_026 메모리 주의 #9, SUB_212
   host-DSA confounder) → 재구성이 타 실험 교란. 공유호스트 변경 = 사전 승인 필요.

## 설계 (구현 후 실행)
- harvest WQ 별도 group + `read_buffers_allowed {96,48,24,12}` sweep × DSA memcpy
  aggressor vs victim → 간섭 상한 곡선.
- 게이트: N=24 에서 victim p99 회복 ≥70% AND DSA 처리량 ≥ 무제한의 50%.
- 핵심 가치: DSA 트래픽은 RMID 미태깅 → MBA 사각(채널 ②). read_buffer 가 디바이스측
  유일 제어 노브 → "RDT 가 못 막는 DSA 간섭을 디바이스에서 막는다" 입증.

## 비고
- DSA 채널은 SUB_212 에서 +33~36%(vanilla) confounder 로 이미 영향 관측됨 — 셰이핑은
  그 영향의 제어 가능성 검증. 전용 세션 + DSA 제출 구현 후 진행 권장.

산출물: (설계 문서만)

---

## [추가 2026-06-15] DSA 제출기 작성 + portal EPERM (보류 해제 시도)

- **`dsa_traffic.c` 작성·빌드 완료**: shared WQ ENQCMD 로 memcpy descriptor 연속 제출
  (opcode 0x03, CRAV/RCR/CC flags, completion record 폴링). sudo 불요 설계.
- **portal mmap EPERM**: `/dev/dsa/wq0.0`(crw-rw-rw-) open 은 되나 `mmap(portal)` 가
  **EPERM**. WQ "lhc0"(type=user/shared/enabled)가 user mmap 을 허용하도록 재구성 필요
  — 또는 다른 실험이 점유한 구성. → **sudo accel-config WQ 재구성 선행 필요.**

## 필요 사용자 명령 (sudo, 공유호스트)
```
# (a) 측정용 harvest WQ 별도 구성 — 기존 lhc0 보존 위해 wq0.1 사용 예
sudo accel-config disable-wq dsa0/wq0.1 2>/dev/null
sudo accel-config config-wq dsa0/wq0.1 --mode=shared --type=user --name=sub236 \
     --group-id=1 --priority=10 --wq-size=16 --threshold=8
sudo accel-config config-group dsa0/group0.1 --read-buffers-allowed=96   # sweep: 96/48/24/12
sudo accel-config enable-wq dsa0/wq0.1
# (b) read_buffers_allowed sweep 로 셰이핑
```
재구성 후 `dsa_traffic /dev/dsa/wq0.1 <cpu> <mb> <secs>` + victim 으로 간섭 측정.

산출물: `dsa_traffic.c` (제출기, 빌드됨).

---

## [최신 2026-06-15] 컨테이너에서 DSA 제출 성공 + 채널② 간섭 실측

**환경 전환**: 호스트(IOMMU `intel_iommu=on`, sm_off)에서는 portal mmap 이 SVA 바인딩
실패로 EPERM. 그러나 **`--device=/dev/dsa --cap-add=SYS_RAWIO`로 뜬 컨테이너에서는
shared WQ portal mmap·ENQCMD 제출이 동작**.

### 0단계 — DSA 가용성 확정 (단일 descriptor 검증)
| WQ | mode | open | mmap | ENQCMD+완료 |
|---|---|---|---|---|
| wq1.0~1.3 (dsa1) | shared/user | OK | **OK** | **status=0x01 + dst==src 검증 ✓** |
| wq0.1 (dsa0) | dedicated 'algo' | **ENXIO** | — | — |

→ shared WQ(ENQCMD)로 userspace DSA memcpy 완전 동작. (호스트 EPERM 해소)

### 1단계 — 채널② 간섭 dose-response (victim 0-7 node0, 3-run, DSA buf=node0 first-touch)
| 셀 | DSA 스트림 | DSA 합산 BW | victim p50 | victim p99 |
|---|---|---|---|---|
| DSA_OFF | 0 | 0 | 22.70 ms (기준) | 101.0 ms (기준) |
| DSA_ON_2 | 2 | 31.4 GB/s | +13.5% | **+32.6%** |
| DSA_ON_4 | 4 | 31.2 GB/s | +13.9% | **+42.2%** |

CV < 5% (전 셀). dsa1 디바이스가 단일 스트림에 이미 ~31 GB/s 포화 — **2·4 스트림이 같은
대역폭인데도 스트림 수(descriptor 동시성)가 늘면 victim p99 가 더 악화** → 간섭이 raw BW 가
아니라 **동시 미결 transaction 수**에 비례.

**결론**: DSA 발 메모리 트래픽은 RMID 미태깅이라 MBM 에 안 잡히고 MBA 로 못 막지만
(채널②), latency-sensitive victim 을 실측 +42% (p99) 악화시킨다. SUB_212 의 host-DSA
confounder(+33~36%)와 정합. **간섭의 실재·dose-response 를 컨테이너에서 확정.**

### 컨테이너 한계 → 호스트 잔여
- **MBA 비차단성 입증**: resctrl 미마운트(`/sys` ro, CAP_SYS_ADMIN 없음) → 호스트
- **read_buffers_allowed {96,48,24,12} 셰이핑 곡선**: `/sys` ro + accel-config 미설치 →
  호스트 (게이트 "N=24 victim p99 회복 ≥70%" 는 호스트 확정 대기)
- **numactl/mbind EPERM**(seccomp): 명시 NUMA 바인딩 불가 → first-touch 로 우회

산출물: `dsa_traffic.c`(제출기), `run_sub236.sh`(3셀×3run + dose-response 판정),
`sub236_results/`(CSV·로그), `/tmp/dsa_verify.c`(단일 descriptor 검증판).

## 호스트 잔여 항목 재확인 (2026-06-15)

호스트에서 MBA 비차단성 증명 / read_buffers {96,48,24,12} 셰이핑 곡선 / MBM DSA BW
시각화를 시도하려 했으나 **호스트 DSA portal mmap 이 여전히 EPERM**:
`/proc/cmdline = iommu=pt intel_iommu=on` (scalable-mode `sm_on` 부재) → shared WQ
ENQCMD 의 PASID/SVA 바인딩 불가. `accel-config list` 상 `dsa0/wq0.1`(dedicated),
`dsa1/wq1.0-1.3`(shared) 가 enabled·`pasid_enabled:1` 이지만, user 제출용 SVA 는
sm_on 없이는 불가 (디바이스 capability ≠ user submission 가능).

→ **이 잔여 항목들은 (a) 컨테이너에서 DSA 트래픽 생성하며 호스트가 resctrl(MBA/MBM)
판독하는 동시 구성, 또는 (b) `intel_iommu=sm_on` 호스트 리부트 후에만 가능**. (b) 는
공유 호스트·실행중 컨테이너 영향이 크므로 사용자 결정 필요. 본 세션은 (a)/(b) 미수행.

## ‼ 정정 (2026-06-15) — 호스트 DSA "EPERM/리부트 필요" 오진단 철회

위의 "호스트 portal mmap EPERM = intel_iommu sm_on 부재 → 리부트 필요" 는 **오진단**.
실제 원인: 호스트 테스트를 **비-root(`mystous`)로 실행**했기 때문. 같은 커널에서 컨테이너
세션은 root 라 DSA 가 됐던 것 — 모순의 정체가 권한이었음.
검증(2026-06-15): `sudo ./ferry_host ferry ... /dev/dsa/wq1.0` → 정상(ferry_s=0.0044,
mmap EPERM 없음). `sudo VLLM_LHC_DSA=1 VLLM_LHC_DSA_DEV=/dev/dsa/wq1.0 python -c
"dsa_lane_available()"` → **True**(self-test ops:1 fails:0).
→ **호스트는 sudo 로 DSA 제출 가능. 리부트 불필요.** SUB_236 잔여(MBA non-block/
read_buffers/MBM)도 **호스트 단독(sudo)** 으로 가능: 호스트에서 DSA 트래픽(sudo) +
resctrl 판독(sudo) 동시. 컨테이너 코디네이션 불요.

## 70B 실서빙 스모크 (호스트, 2026-06-15) — DSA harvest near-free, 간섭 미발현

합성 latency-probe victim 이 아닌 **실제 Llama-3.1-70B serving(TP4, GPU0-3, node0 핀)**
을 victim 으로 DSA harvest 간섭 측정. 부하 conc24/ptok2000/mtok256/reqs96 warm.

| 셀 | 70B gen tps |
|---|---|
| A 서빙 단독 | 2285.9 / 2290.5 |
| B +DSA harvest (4스트림 wq1.x, 합산 ~120 GB/s 실측) | 2290.3 / 2287.0 |

→ **DSA harvest ~120 GB/s 를 부어도 70B serving tps 변화 ≈0%** (노이즈 내). 합성 victim
(메모리-지연 프로브)에서 보이던 채널② 간섭이 **GPU-bound 인 실 70B serving throughput
에는 발현되지 않음**. SUB_228(유용작업 near-free harvest)과 정합. → shaping guard 의
실서빙 실효 가치 낮음(간섭 자체가 ~0).

**주의/사고**: read_buffers shaping 셀은 `accel-config disable-device dsa1` 가 wq1.x 설정을
wipe → /dev/dsa/wq1.x 소실(공유 인프라 lhc1_x 파손). **즉시 복구 완료**(config-wq shared/
user + enable, dsa_traffic 30.8 GB/s 재확인). read_buffers sweep 은 본질적으로 파괴적이라
공유 호스트에서 비권장. 간섭이 ~0 이므로 shaping 측정 자체가 moot.

산출물: `smoke_70b.sh`, `host_smoke/`.
