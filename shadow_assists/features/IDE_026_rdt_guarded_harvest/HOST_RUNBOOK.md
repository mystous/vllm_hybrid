# IDE_026 / TSK_048 — 호스트 Runbook (T0 + T1)

> **전제 (2026-06-11 확정)**: 본 컨테이너는 비특권 (`CAP_SYS_ADMIN` 없음 → resctrl mount
> EPERM) + device cgroup 이 `/dev/cpu/N/msr` open 을 차단 (EPERM, 사용자 실측).
> **컨테이너 내 RDT 제어 불가** — 아래 작업은 **호스트에서 root 로** 실행한다.
> `/workspace` 가 호스트와 공유 볼륨이면 동일 경로에서 그대로 실행 가능.

## 0. 사전 확인 (호스트, 1분)

```bash
# resctrl 지원 플래그 (이미 컨테이너에서 확인됨: cat_l3 cdp_l3 mba cqm_mbm_total 등)
grep -o 'cat_l3\|mba\|cqm_mbm_total' /proc/cpuinfo | sort -u

# mount (재부팅 전까지 유지; 이미 마운트면 no-op)
sudo mount -t resctrl resctrl /sys/fs/resctrl

# capability 확인
cat /sys/fs/resctrl/info/L3/cbm_mask        # 예: 7fff → 15 ways
cat /sys/fs/resctrl/info/MB/min_bandwidth   # 보통 10
cat /sys/fs/resctrl/info/L3_MON/mon_features
```

## 1. T0 — harness self-test (G0, 1분)

```bash
cd /workspace/host_vllm_hybrid/shadow_assists/features/IDE_026_rdt_guarded_harvest/src
sudo python3 rdt_ctl.py selftest
# 기대: [selftest] ALL PASS
#   - dual-socket 도메인 2개 (0=, 1=)
#   - schemata 기록→재판독 일치 / TID 등록 확인 / mbm 단조성
sudo python3 rdt_ctl.py info   # cbm 폭·min_bandwidth 를 MEASUREMENTS 에 기록
```

## 2. T1 — CPU-only 간섭 재현 + CAT/MBA A/B (G1, 약 25분)

```bash
cd /workspace/host_vllm_hybrid/shadow_assists/features/IDE_026_rdt_guarded_harvest/src
bash build.sh                  # gcc -O3 -march=native; "zmm 명령 포함 확인" 떠야 정상
sudo bash run_t1_ab.sh         # 4셀(IDLE/B0/B1/B2) × 3-run × 30s ≈ 25분
```

- 결과: `src/t1_results/t1_summary.csv` + 셀별 `mon_*.csv` (llc_occupancy / mbm GB/s 시계열)
- 스크립트 말미에 **G1 게이트 자동 판정** 출력:
  - 간섭 실재: B0 p99 ≥ IDLE p99 +10%
  - L2 GO: B2 회복 ≥ 80% AND B2 aggressor ≥ B0 의 70%
  - 사전 예측 (test.md G5): B0 악화 +15~60%, B2 회복 80~95%
- 기본 배치: victim `0-7`, aggressor `16-55` — **둘 다 socket 0** (MBA 는 socket-로컬
  트래픽만 throttle 하므로 cross-socket 배치는 B2 효과가 희석됨).
  task.md 원안 (16-111, 양 socket) 비교가 필요하면:
  `sudo AGGR_CPUS=16-111 OUT=./t1_results_xsock bash run_t1_ab.sh`
- 간섭이 안 보이면 (FAIL): aggressor 강도 1회 재설계 한도 —
  `sudo VICTIM_WS=… AGGR_CPUS=8-55 bash run_t1_ab.sh` 식으로 코어 추가 또는
  `--array-mb` 증가 (run_t1_ab.sh 내 BIN 호출부 수정).

## 3. (선택) 이후 컨테이너에서 계속 작업하려면 — 옵션 (a)

컨테이너 재기동 시 docker run 인자에 추가:

```bash
-v /sys/fs/resctrl:/sys/fs/resctrl        # resctrl 통째 bind (필수)
# 선택 (MSR fallback 까지 열려면):
--cap-add SYS_ADMIN                       # 또는 device cgroup: --device-cgroup-rule 'c 202:* rwm'
```

이후 T2~T4 (vLLM 스레드 MBM attribution, L2/L3 통합) 를 컨테이너 내에서 진행 가능.
재기동이 곤란하면 옵션 (b): T1 까지는 위처럼 호스트 실행, T2 는 GPU 가용 시점에 재논의.

## 4. 결과 회수

`t1_results/` 는 공유 볼륨 경로에 생성되므로 컨테이너에서 바로 보인다.
회수 후 컨테이너 쪽 작업: G1 판정 기록 → `MEASUREMENTS.md` 작성 → T2 준비.
