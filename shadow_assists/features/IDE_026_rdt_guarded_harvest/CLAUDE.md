# IDE_026 — CLAUDE.md (구현 시 알아야 할 것)

## ✅ 환경 전환 (2026-06-12): 호스트 직접 작업 — 아래 컨테이너 섹션은 이력

- 현재 작업 환경 = **dgx-b200 호스트** (mystous + sudo). 컨테이너 권한 문제 전부 해소.
- **resctrl 이미 마운트됨** (`/sys/fs/resctrl` rw, 커스텀 그룹 0개 — 본 작업이 첫 사용자).
  mount 단계 불필요. T0 selftest 부터 즉시 실행 가능: `sudo python3 src/rdt_ctl.py selftest`
- HW capability 실측 완료 (2026-06-12) — **task.md T0 표** 와 `RESEARCH_DIRECTIONS.md` §1 참조.
  요점: L3 CAT 20-way/15CLOS (way=15MB, 18-19 는 IO 공유), **L2 CAT 16-way/8CLOS 지원**,
  MBA 10%/linear/`thread_throttle_mode=max`, MBM total+local (480 RMID), waitpkg/cldemote ✓.
- vLLM 실행 env 는 메모리 `host-vllm-build-env` 참조 (CUDA_HOME=/usr/local/cuda-13.0 필수).

## ⚠ 컨테이너 권한 확정 사항 (2026-06-11 직접 확인 — 이력, 호스트 전환으로 해소)

- 본 작업 환경은 **비특권 컨테이너**: `CapEff=0xa80665fb` = Docker 기본 +
  `CAP_IPC_LOCK`(14) + `CAP_SYS_RAWIO`(17). **`CAP_SYS_ADMIN`(21) 없음**
  → **컨테이너 내 `mount -t resctrl` 불가 (EPERM)**. resctrl 은 namespace
  가상화도 없어 컨테이너 시작 후 호스트 마운트는 전파되지 않음 (기본 private).
- **우회 경로 = MSR 직접 프로그래밍** (pqos MSR-모드와 동일 방식):
  - 호스트 msr 드라이버 로드됨 (`/proc/devices` 에 `202 cpu/msr`) ✓
  - `CAP_MKNOD`(27) 있음 → `mknod /dev/cpu/N/msr c 202 N` 가능 ✓
  - `CAP_SYS_RAWIO` 있음 (msr read/write 의 커널 요구 cap) ✓
  - **시험 결과 (2026-06-11, 사용자 실행): 기각** — `mknod` 는 성공했으나
    `open('/dev/cpu/0/msr', O_RDONLY)` 가 **EPERM** = device cgroup
    whitelist 차단. **컨테이너 내 RDT 제어는 resctrl·MSR 두 경로 모두 불가 확정.**
- MSR 모드 설계 변경점: CLOS 바인딩이 **task 단위가 아닌 코어 단위**
  (`IA32_PQR_ASSOC`=0xC8F per-logical-CPU). 우리 설계는 어차피
  serving/harvest 코어를 affinity 로 분리하므로 코어 단위로 충분.
  - CAT mask: `IA32_L3_QOS_MASK_n` (0xC90+n), MBA: `IA32_MBA_THRTL_MSR`
    (0xD50+n), CMT/MBM 판독: `IA32_QM_EVTSEL`(0xC8D)+`IA32_QM_CTR`(0xC8E).
- **확정 결론: 호스트 측 1회 작업 필수.** 두 가지 택일:
  - (a) **호스트에서** `mount -t resctrl resctrl /sys/fs/resctrl` 실행 후
    컨테이너 재기동 시 `-v /sys/fs/resctrl:/sys/fs/resctrl` bind-mount
    (+ 가능하면 `--cap-add SYS_ADMIN` 또는 device cgroup 에 `c 202:* rwm` 추가
    — 후자는 MSR fallback 용 선택사항). → 이후 모든 T0~T4 를 컨테이너에서 진행.
  - (b) 컨테이너 재기동이 곤란하면: T1 (CPU-only victim/aggressor A/B) 을
    **호스트에서 직접** 실행. 코드·스크립트는 컨테이너에서 작성하고
    `/workspace` 가 호스트와 공유 볼륨이면 호스트 쪽에서 같은 경로로 실행.
- **컨테이너 내에서 지금 가능한 것** (RDT 권한과 무관) — **2026-06-11 작성 완료**:
  `src/victim_aggressor.c` + `src/build.sh` (마이크로벤치), `src/rdt_ctl.py`
  (파일 I/O 추상화 — `--root` 인자로 호스트에서도 동일 동작),
  `src/run_t1_ab.sh` (4셀 A/B + G1 자동 판정), `HOST_RUNBOOK.md`.
  빌드 검증은 Bash 분류기 불가로 보류 — 호스트/로컬에서 `bash src/build.sh`.

## resctrl 사용법 (호스트/bind-mount 환경 — 외부 라이브러리 불필요)

```bash
# 1. mount (root 1회)
mount -t resctrl resctrl /sys/fs/resctrl

# 2. HW capability 확인
cat /sys/fs/resctrl/info/L3/cbm_mask      # 예: fffff (way bitmask 폭)
cat /sys/fs/resctrl/info/L3/min_cbm_bits
cat /sys/fs/resctrl/info/MB/min_bandwidth # MBA 최소 % (보통 10)
cat /sys/fs/resctrl/info/L3_MON/mon_features  # llc_occupancy mbm_total_bytes mbm_local_bytes

# 3. CLOS 그룹 생성 + 스키마
mkdir /sys/fs/resctrl/serving /sys/fs/resctrl/harvest
echo "L3:0=ffff0;1=ffff0" > /sys/fs/resctrl/serving/schemata   # 상위 way
echo "L3:0=0000f;1=0000f" > /sys/fs/resctrl/harvest/schemata   # 하위 4 way
echo "MB:0=100;1=100"     > /sys/fs/resctrl/serving/schemata   # (같은 파일에 두 줄로 써도 됨)
echo "MB:0=20;1=20"       > /sys/fs/resctrl/harvest/schemata

# 4. 스레드(TID) 등록 — PID 가 아니라 TID 단위 가능
echo <tid> > /sys/fs/resctrl/harvest/tasks

# 5. MBM/CMT 판독 (그룹별·도메인별)
cat /sys/fs/resctrl/serving/mon_data/mon_L3_00/llc_occupancy
cat /sys/fs/resctrl/harvest/mon_data/mon_L3_00/mbm_total_bytes
```

- **schemata 는 socket(도메인)별** — 8570 dual socket 이므로 `0=`, `1=` 두 도메인.
- **CDP 활성 시** (`mount -o cdp`) L3CODE/L3DATA 분리 — 1차는 CDP 없이.
- way overlap 허용됨 (serving ⊃ harvest 도 가능) — 1차는 **비중첩**으로 단순하게.
- 컨테이너에서 /sys/fs/resctrl 가 안 보이면 host 측 마운트 필요 (R3) —
  사용자에게 `! sudo mount -t resctrl resctrl /sys/fs/resctrl` 안내.

## Python 측 통합 지점 (이 fork 의 기존 hook 재사용)

- `vllm/v1/worker/gpu_worker.py:191-202` — NEO pinning 의 `sched_setaffinity` 자리:
  같은 위치에서 `threading.get_native_id()` 를 resctrl tasks 에 echo 하는 helper 추가.
- `vllm/v1/lhc/metronome/meter.py:69`, `tempo.py:175` — lhc-tempo 스레드 → harvest CLOS.
- `vllm/v1/spec_decode/ngram_proposer.py:68` — precompute pool → harvest CLOS
  (`ThreadPoolExecutor(initializer=...)` 로 TID 등록).
- env 네이밍: `VLLM_RDT_ENABLE`, `VLLM_RDT_HARVEST_WAYS`, `VLLM_RDT_HARVEST_MBA`
  (+ flag 파일 fallback — SUB_213 의 `_env_bool` 패턴 재사용, `suffix_decoding.py:88`).

## tpause duty-cycle (L3 governor)

- `waitpkg` 확인 완료. C0.2 (느린/저전력) state: `tpause(TSC_deadline, ctrl=0)`.
- Python 에서 직접 불가 → ctypes 로 작은 .so (`_mm_tpause` intrinsic) 또는
  기존 csrc/cpu/ 빌드 체계에 1 함수 추가. dev 머신 (12900KF, Alder Lake) 도
  waitpkg 지원이므로 단위 테스트 가능.

## 측정 도구

- MBM 카운터 오버플로: EMR 은 62bit — 사실상 무시 가능, 그래도 delta 방식으로 판독.
- llc_occupancy 는 절대값 (bytes), mbm_* 는 누적 bytes → 주기 샘플링 (lhc-tempo 패턴 재사용).
- mpstat/turbostat 병행 (CPU util Objective 검증). perf 부재 환경 (TSK_023 교훈) —
  resctrl 판독은 perf 불필요 (파일 read 만).

## 함정

1. **tasks 등록은 자식 스레드에 상속 안 됨** — 스레드 생성 *후* TID 별 등록 필요.
   ThreadPoolExecutor 는 `initializer=` 콜백에서 자기 TID 를 등록하는 패턴이 정석.
2. CLOS 수 제한 (실측): **L3=15, L2=8, MB=15** — L2 CAT 병용 시 8 이 바인딩 상한
   (closid 는 자원 간 공유). 그룹 2~3개면 충분.
3. MBA 는 **socket-로컬 트래픽만** throttle — cross-socket (UPI) 은 제어 밖.
   NUMA bind (기존 N8/SUB_165 지식) 와 병행해야 의미. 원격 비율은
   `mbm_total_bytes − mbm_local_bytes` 로 정량 (RESEARCH_DIRECTIONS.md D6).
4. 측정 중 cudagraph/FaP 모드 고정 (SUB_212 confounder 교훈).
5. GPU 점유 중 — (a)(b) 단계만 진행, vLLM 부팅 금지.
6. **`thread_throttle_mode=max` (실측)** — SMT sibling 두 스레드가 다른 CLOS 면
   **가장 강한 MBA throttle 이 물리 코어 전체에 적용**. serving 과 harvest 가 같은
   물리 코어의 HT 짝으로 들어가면 serving 이 연좌 throttle 됨 → MBA 사용 시
   **코어-배타 배치 필수** (D1 에서 정량화).
7. **CBM 은 연속 비트만 허용 (실측 `sparse_masks=0`)** — `0x000f0` OK, `0x000f1` EINVAL.
8. **L3 way 18-19 는 IO(DDIO) 공유 (실측 `shareable_bits=c0000`)** — GPU PCIe DMA
   착지 way. harvest mask 에 포함 금지 (D3). serving mask 에는 포함해도 무방.
9. **RDT 불가시 트래픽** — DSA/IAA engine 트래픽과 GPU DMA 는 RMID 미태깅 (MBM 에
   안 잡힘) + MBA 미적용. SUB_212 의 host-DSA confounder 가 실례. 제어는 WQ 구성
   /제출률/duty-cycle 로만 가능 (RESEARCH_DIRECTIONS.md §2 분류학).
10. resctrl 그룹 삭제는 `rmdir` — 그룹에 등록됐던 TID 는 자동으로 루트 그룹 복귀.
    실험 후 `teardown` 으로 원상복구 (이미 마운트된 자원을 umount 하지 말 것 —
    다른 사용자가 쓸 수 있음).
