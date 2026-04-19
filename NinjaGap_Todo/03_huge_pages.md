# 03. Huge Pages (2MB THP / 1GB hugetlb)

**Tier**: 0
**상태**: ✗ **기각 (2026-04-19)** — Phase 1 호스트 default 로 이미 활성, Phase 2 는 역효과
**예상 이득 (원 계획)**: 2MB THP **3–10%** · 1GB 추가 **3–10%**
**실제 검증 결과**: Phase 1 은 별도 작업 불필요 (호스트 THP=always 가 이미 기본). Phase 2 (1GB) 는 H100x8 + Qwen2.5-32B 실측에서 **baseline 대비 +22% 더 느림** (역효과)

---

## 기각 근거 (2026-04-19 H100x8 실측)

### Phase 1 (2MB THP)

호스트 RHEL 에서 `/sys/kernel/mm/transparent_hugepage/enabled` 의 기본값이 이미 `always`. 실측 시 `/proc/meminfo`:

```
AnonHugePages:   4216832 kB     # 4.1 GB 이미 2MB 페이지로 승격 상태
Hugepagesize:       2048 kB
```

즉 기존 baseline 측정이 이미 2MB THP 적용 상태에서 이루어짐 → **별도 작업으로 얻을 gain 0**. 이전 제안 값 (3–10%) 은 THP off 가정이었으나 환경은 이미 on.

### Phase 2 (1GB hugetlbfs)

`feat/g0-03-phase2-1gb-hugetlb` branch 에 구현 후 H100x8 (Xeon 8480+ 2-socket) × Qwen2.5-32B 에서 실측:

| 실험 | hugetlb 적용량 | wall duration | baseline 대비 |
|---|---:|---:|:---:|
| baseline (no hugetlb, 2MB THP 만) | 0 GiB | **67.9 s** | 기준 |
| 부분 적용 (풀 64 GiB 제한) | 2×30 GiB | 78~79 s | **+15 ~ 16%** |
| 거의 완전 적용 (풀 128 GiB, memlock=-1) | 2×58 GiB | **82~83 s** | **+22%** |

- `migrated=446 (58.00 GiB), skipped=5 (3.03 GiB)` per CPU engine — 모델 weight 거의 전부 1GB 페이지로 이동
- CPU_EngineCore_1 / _2 각각에 동일 로그 확인 — 양 NUMA 노드 모두 적용
- 적용 범위가 커질수록 **monotonic 하게 더 느려짐** — 측정 노이즈 아닌 구조적 역효과

### 역효과의 원인 — SPR TLB 구조

Xeon 8480+ (Sapphire Rapids) 의 L2 dTLB 구조:

| 페이지 크기 | L2 dTLB entries | 총 coverage |
|---|---:|---:|
| 4KB + 2MB (통합) | **2048** | 4KB×2048 = 8 MiB 또는 2MB×2048 = **4 GiB** |
| 1GB (별도) | **16** | 1GB×16 = **16 GiB** (표면 수치) |

원 계획은 "1GB 16 × 1GB = 16 GiB coverage 가 2MB 4 GiB 보다 큼" 이었으나 실제로는:

- **1GB 페이지는 완전 별도 TLB 를 쓰고, 2048 entries 의 큰 pool 을 전혀 못 씀**. 즉 1GB 페이지로 옮기면 L2 dTLB 의 주력 pool (2048 × 2MB = 4 GiB) 을 버리고 16 entry 짜리 작은 pool 로 바꾸는 셈.
- 32B weight ≈ 64 GiB / 1GB = **64 pages** 인데 1GB TLB 는 16 entries 만 유지 → 대부분 접근에서 **L2 TLB miss + 4-level page walk**.
- 반면 2MB THP 는 `named_parameters` 대부분이 수백 MB 이하라 hot layer 의 weight 이 2MB 단위 → 2048 × 2MB = 4 GiB coverage 가 hot working set 에 충분.

따라서 **이 workload (32B BF16, TP 기반 hybrid) 에서는 2MB THP 가 sweet spot** 이고 1GB 로 내려가면 page walk 가 오히려 증가.

### KV cache 는 hugetlb 에 들어가지도 못함

풀 128 GiB 확장 후에도 weight 에 ~58 GiB × 2 = 116 GiB 가 먼저 소비되어 KV cache allocator 가 mmap 에서 ENOMEM. 풀을 더 크게 확보해도 weight 쪽의 역효과가 지배적이라 시도 의미 없음.

---

## 결론

Phase 1 은 "호스트 default 로 이미 on 이라 별도 작업 불필요" 로 종결. Phase 2 는 SPR 의 1GB dTLB 작음 + 2MB 공유 TLB 포기라는 구조적 이유로 **더 큰 페이지가 역효과**. 두 Phase 모두 Ninja Gap 기여 없음.

→ **§03 전체 기각**. CPU path 의 성능 이득은 TLB 구간이 아니라 kernel / batch scaling 구간에서 찾아야 함 (§06 이후).

---

## 남겨진 산출물 (히스토리 보존)

- `feat/g0-03-phase2-1gb-hugetlb` branch (force-push / rebase 금지, 기록 유지)
- 그 branch 안 주요 코드:
  - `vllm/platforms/hugetlb_allocator.py` — HugeTLB1GAllocator + slab sub-allocator
  - `vllm/v1/worker/cpu_model_runner.py` — KV cache + post-IPEX weight bind hook
  - `eval/envs/g0_h100x8_qwen32b_hugetlb_1g_{kv,full}.env` — 측정 env
  - `setup/setup_hugetlb_1g_rhel.sh` — 호스트 풀 관리 도구 (enable/disable/verify)
  - `setup/verify_hugetlb_1g.sh` — 현장 end-to-end 검증 script
- 실측 결과: `eval/results/20260419_02*_*_seqs1/`, `20260419_03*_*_seqs1/`

Phase 1 Infra 는 main 에 이미 존재 (`setup/setup_thp_rhel.sh`, `eval/envs/g0_h100x8_qwen32b_thp.env`). THP=always 기본값 확인 용도로 유지.

---

## 왜 필요한가

70B INT4 모델 기준:
- 4KB 페이지 사용 시 TLB 엔트리 **9,000,000 개** 필요
- 1GB 페이지 사용 시 TLB 엔트리 **35 개** — Xeon 8480+ DTLB 엔트리 수 (64 × 2MB, 32 × 1GB) 에 여유 있게 수용
- TLB miss → page walk 비용 (L1 TLB miss ~10 cycles, L2 TLB miss ~30 cycles, page walk ~100-300 cycles) 이 decode 전반에 누적

Linear weight 와 KV cache 모두 장기 resident → **hugetlbfs 적합도 가장 높은 대상**.

---

## 기술적 배경

### 페이지 크기 계층

여기서 말하는 것은 **캐시 크기**가 아니라 **페이지 크기**다. Linux/x86 에서
실무적으로 의미 있는 기본 선택지는 거의 다음 셋뿐이다.

| 크기 | Xeon 8480+ TLB 엔트리 | 사용 대상 |
|---|---|---|
| 4KB | L1 64 (unified), L2 2048 (shared with 2MB) | 기본 |
| 2MB | L1 32 (data), L2 공유 | Transparent Huge Pages (THP) |
| 1GB | L1 4 (data), L2 16 | `hugetlbfs` explicit |

즉, "2MB 다음 중간 크기"가 빠진 것이 아니라 **OS / 하드웨어가 보편적으로
제공하는 표준 페이지 크기 자체가 4KB → 2MB → 1GB 로 듬성듬성하다**고
보는 편이 정확하다.

### 왜 2MB와 1GB를 분리해서 봐야 하나

- **2MB THP**:
  - 커널이 anonymous memory 를 자동 promotion 할 수 있다
  - 재부팅 없이 시도 가능
  - 코드 수정 없이도 효과를 볼 수 있다
  - 따라서 **가장 먼저 닫아야 할 단계**

- **1GB hugetlb**:
  - 커널이 자동으로 승격해주지 않는다
  - reserve / mount / explicit mmap 이 필요하다
  - 장기 uptime 서버에서는 단편화 때문에 실패할 수 있다
  - 따라서 **2MB 이득이 확인된 뒤의 추가 단계**

### 효과 비교 (모델 크기별)

| 모델 weight | 4KB 커버율 (L2 TLB 2048 기준) | 2MB | 1GB | 2MB→1GB 추가 이득 |
|---|---|---|---|---|
| 1.5B (3GB) | 0.3% | 100% | 100% | 거의 없음 (2MB 충분) |
| 7B (14GB) | 0.06% | 29% (hot set OK) | 100% | 3–10% |
| 32B (64GB) | 0.01% | 6% | 100% | 5–10% |
| 70B INT4 (35GB) | 0.02% | 11% | 46% | 10%+ 가능 |

**4KB → 2MB 가 먼저 확인해야 할 구간**이고, 2MB → 1GB 는 그 다음의
추가 최적화다. 7B 에서는 1GB 가 "압도적 필수"라기보다 **운영 복잡도를
감수할 만큼의 추가 이득이 있는지 확인해야 하는 단계**다.

### RHEL 기본 상태

- `cat /sys/kernel/mm/transparent_hugepage/enabled` → `always [madvise] never`
- 기본 `madvise` = `madvise(MADV_HUGEPAGE)` 힌트 있을 때만 THP. vLLM 은 hint 안 주므로 **실질 효과 없음**. `always` 로 전환 필요.

### 구현 경로 3가지

**경로 A — 2MB THP, 재부팅 없이 (권장, RHEL)**:

즉시 적용 (sudo):
```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag
```
영구 적용 (재부팅 없이) — systemd:
```bash
cat > /etc/systemd/system/thp-always.service <<'EOF'
[Unit]
Description=Transparent Huge Pages = always
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'
ExecStart=/bin/sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/defrag'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now thp-always
```
또는 tuned (RHEL 표준):
```bash
mkdir -p /etc/tuned/vllm-hybrid
cat > /etc/tuned/vllm-hybrid/tuned.conf <<'EOF'
[main]
include=throughput-performance
[vm]
transparent_hugepages=always
EOF
tuned-adm profile vllm-hybrid
```

**vLLM 코드 변경 불필요** (THP 는 커널 자동 promotion). 설정 후 vLLM **재시작**
만 필요. 따라서 이 경로가 사실상 "Huge Pages의 1차 실험"이다.

**경로 B — 1GB 런타임 할당, 재부팅 없이**:

메모리 compact + 1GB 페이지 요청 (sudo):
```bash
echo 1 > /proc/sys/vm/compact_memory
echo 40 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
# 실제 확보 확인
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
cat /proc/meminfo | grep -E 'HugePages_Total|HugePages_Free'
```
NUMA 노드별:
```bash
echo 20 > /sys/devices/system/node/node0/hugepages/hugepages-1048576kB/nr_hugepages
echo 20 > /sys/devices/system/node/node1/hugepages/hugepages-1048576kB/nr_hugepages
```

장기 uptime 서버는 단편화로 **부분 확보 (12/40 등) 또는 0 확보**가 흔하다.
즉 1GB 는 "중간 단계"가 아니라 **운영 난이도가 급격히 올라가는 별도 단계**다.

**+ vLLM 코드 수정 필수** (THP 자동 promotion 은 2MB 까지만, 1GB 는 explicit mmap 필요):
```c
mmap(NULL, size, PROT_READ|PROT_WRITE,
     MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|(30 << MAP_HUGE_SHIFT),
     -1, 0);  // 30 = log2(1GB)
```

**경로 C — 1GB boot-time 할당 (재부팅 필수)**:

가장 확실한 방법. 운영팀 협의 필요.
```bash
# /etc/default/grub 에 추가
GRUB_CMDLINE_LINUX="... hugepagesz=1G hugepages=40 default_hugepagesz=1G"
grubby --update-kernel=ALL --args="hugepagesz=1G hugepages=40"
# 재부팅 후
mkdir -p /dev/hugepages-1G
mount -t hugetlbfs -o pagesize=1G none /dev/hugepages-1G
```
+ vLLM 코드 수정 (경로 B 와 동일).

### 컨테이너 vs 호스트 권한

- `/sys/kernel/mm/...` 는 **호스트에서만 쓰기 가능**. 컨테이너는 read-only mount.
- 호스트에서 설정 → 모든 컨테이너가 자동으로 공유.
- 확인: 컨테이너 안에서 `cat /sys/kernel/mm/transparent_hugepage/enabled` 로 상태 확인 가능.
- 프로세스가 실제 THP 쓰는지: `grep AnonHugePages /proc/$(pgrep -f api_server)/smaps | awk '{s+=$2} END {print s/1024 "MB"}'`

### 현재 vLLM 상태

- `grep MAP_HUGETLB` — 결과 없음 (explicit 1GB mmap 사용 안 함)
- `grep hugepagesz` — 결과 없음
- weight mmap 은 `safetensors` 경유 → page fault on demand (4KB)
- IPEX oneDNN scratchpad 는 malloc → THP 범주에 들어옴

---

## 관련 참고 문헌

- **Intel Optimization Reference Manual** §3.6 "Memory Subsystem Optimization": https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
- **Red Hat Performance Tuning Guide — HugeTLB**: https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/performance_tuning_guide/sect-red_hat_enterprise_linux-performance_tuning_guide-memory-configuring-huge-pages
- **deep-research report §"남은 불확실성 4"**: `/vllm_hybrid/ideation/20260415_1629_deep-research-report.md` — 운영 제약 언급
- **Basu et al. (2013) "Efficient Virtual Memory for Big Memory Servers"** (ISCA): TLB miss overhead 정량화
- **LWN "Huge pages part 1: Introduction"**: https://lwn.net/Articles/374424/
- **Linux kernel `Documentation/admin-guide/mm/hugetlbpage.rst`**: https://www.kernel.org/doc/html/latest/admin-guide/mm/hugetlbpage.html

---

## 구체 작업

### Phase 1 — 경로 A (2MB THP, 재부팅 없이) — **먼저 시도**

- [ ] **호스트 (RHEL) 에 THP=always 적용**: systemd 또는 tuned 중 선택
- [ ] **현재 상태 baseline 측정**: `cat /sys/kernel/mm/transparent_hugepage/enabled` → `madvise` 상태에서 `serve.sh + bench.sh` 로 wall/TPOT 기록
- [ ] **THP always 전환 후 vLLM 재시작** → 재측정
- [ ] **프로세스 실제 사용량 확인**: `grep AnonHugePages /proc/$(pgrep -f api_server)/smaps` 로 수 GB 이상 확인
- [ ] **perf 로 dTLB miss 비교**: `perf stat -e dTLB-load-misses,dTLB-load-miss-walk-pending -p $PID sleep 30`
- [ ] **G0 sweep 재실행**: `measurement_results/<HW>/g0_03_thp2m/seqs<N>/` 로 분리 저장

**Phase 1 성공 시 해석**:
- THP 만으로도 의미 있는 개선이 나오면, TODO 본문에서는 이를 `Huge Pages`의
  기본 완료로 간주할 수 있다.
- 그 이후 1GB 는 "필수 선행"이 아니라 **추가 ROI 실험**이다.

### Phase 2 — 경로 B (1GB 런타임, 재부팅 없이) — **Phase 1 이득 확인 후**

- [ ] **1GB 페이지 확보 시도**: `echo 40 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages` + NUMA 노드별 분배. 실제 확보량 로그.
- [ ] **vLLM weight loader 수정**: `vllm/model_executor/model_loader/loader.py` 에 `HYBRID_HUGEPAGES=1gb` 분기. `torch.from_file` + 수동 `mmap(MAP_HUGETLB | MAP_HUGE_1GB)`.
- [ ] **KV cache allocator 수정**: `vllm/v1/worker/cpu_worker.py` 의 `VLLM_CPU_KVCACHE_SPACE` 참조부에 1GB mmap 경로 추가.
- [ ] **Fallback chain**: 1GB 실패 → 2MB (MAP_HUGE_2MB) → 4KB. `[HYBRID-HUGEPAGE] page_size=? weight_bytes=? kv_bytes=?` 로그 marker.
- [ ] **G0 sweep 재실행**: `measurement_results/<HW>/g0_03_1g/seqs<N>/` 로 분리 저장. Phase 1 (2MB) 대비 추가 이득 측정.

**Phase 2 진입 조건**:
- 2MB THP 에서 이득이 명확할 것
- 대상 workload 가 memory / TLB bound 라는 정황이 있을 것
- 운영 복잡도 증가를 감수할 이유가 있을 것

### Phase 3 — 경로 C (1GB boot-time, 재부팅 필요) — **운영팀 승인 시**

- [ ] **운영팀에 재부팅 창 요청** (Phase 2 에서 의미 있는 추가 이득 확인 시에만)
- [ ] **GRUB 파라미터**: `grubby --update-kernel=ALL --args="hugepagesz=1G hugepages=40"`
- [ ] **재부팅 후 hugetlbfs mount**: `/dev/hugepages-1G`
- [ ] **재측정**: runtime 대비 확보 안정성 개선 확인

---

## 성공 조건

### Phase 1 (2MB THP)
- `grep AnonHugePages /proc/$(pgrep -f api_server)/smaps` → **수 GB 이상** (weight 대부분 커버)
- `perf stat -e dTLB-load-misses` 기준 dTLB miss **80% 이상 감소**
- decode per-step 시간 **3–10% 감소** (7B 기준 보수적)
- PPL 변화 없음

### Phase 2–3 (1GB)
- Phase 1 대비 **추가 3–10% decode 단축** (7B), **10%+** (70B)
- `HugePages_Total` 에 요청량 (Phase 2) 또는 40 (Phase 3) 확보
- PPL 변화 없음

### 조기 중단 기준
- Phase 1 이득 < 3% → Phase 2 진행 안 함 (ROI 낮음, 다른 기법 우선)
- 7B 에서 2MB 로 이미 hot set miss 가 충분히 줄었다고 보이면, 1GB 는 문서상
  "보류"로 남겨도 된다

---

## 의존성

- **선행**: §01 G0 계측 (baseline 확보)
- **병행**: §04 IPEX WoQ INT8, §05 OMP env
- **후속**: 모든 기법 (항상 깔려 있어야 하는 infra)

---

## 리스크

- **호스트 sudo 권한 필요**: `/sys/kernel/mm/` 쓰기는 호스트 root 전용. 컨테이너에서는 read-only. 운영팀과 협의해 호스트 세팅 적용.
- **1GB 런타임 확보 실패**: 장기 uptime 서버의 메모리 단편화로 부분/전체 실패. 대처: 2MB THP 로 후퇴, 또는 Phase 3 boot-time.
- **HuggingFace/safetensors 가 mmap path 에 hardcoded**: 1GB 쓰려면 `torch.from_file` + 수동 mmap 경로 구축 필요 (non-trivial). 2MB THP 경로는 이 수정 불필요.
- **이득 측정 어려움**: 다른 기법 (WoQ INT8, ISA dispatch) 과 동시 적용 시 단독 기여도 분리 필요. **Phase 별로 단독 측정**.
- **khugepaged jitter**: THP 백그라운드 promotion 데몬이 짧은 stall 유발 가능. 발견 시 `defrag=defer+madvise` 로 조정.

---

## 스택 호환성

**모든 후속 기법과 독립**. 항상 깔려 있어야 함. Huge Pages 로 확보한 TLB 여유는 §15 AMX pre-pack, §13 T-MAC LUT 등 memory-bound 기법의 이득 상한을 높임.

---

## 실행 flag

| flag | 값 | 의미 |
|---|---|---|
| `VLLM_HYBRID_PROFILE=1` | 측정 모드 | manifest + sublayer hook 활성 |
| `HYBRID_HUGEPAGES` | `0` (기본) / `1` | 본 기법 활성 |

전체 flag 테이블: [README.md](./README.md) "기법 Feature Flag 테이블" 참조.

---

## 관련 코드 위치

- `vllm/model_executor/model_loader/loader.py` — weight loading
- `vllm/v1/worker/cpu_worker.py` — `VLLM_CPU_KVCACHE_SPACE` 참조
- `vllm/v1/core/kv_cache_manager.py` — KV allocation
- `eval/envs/h100x8_*.env` — `VLLM_CPU_KVCACHE_SPACE` 설정
