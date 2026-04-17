# 03. Huge Pages (2MB / 1GB)

**Tier**: 0
**상태**: ⭕ 미구현
**예상 이득**: 2MB THP **10–30%** · 1GB 추가 **3–10%** (7B 기준)
**권장 순서**: Phase 1 (2MB THP, 재부팅 없이) → 이득 확인 후 Phase 2 (1GB 런타임) → 필요 시 Phase 3 (재부팅)

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

| 크기 | Xeon 8480+ TLB 엔트리 | 사용 대상 |
|---|---|---|
| 4KB | L1 64 (unified), L2 2048 (shared with 2MB) | 기본 |
| 2MB | L1 32 (data), L2 공유 | Transparent Huge Pages (THP) |
| 1GB | L1 4 (data), L2 16 | `hugetlbfs` explicit |

### 효과 비교 (모델 크기별)

| 모델 weight | 4KB 커버율 (L2 TLB 2048 기준) | 2MB | 1GB | 2MB→1GB 추가 이득 |
|---|---|---|---|---|
| 1.5B (3GB) | 0.3% | 100% | 100% | 0 (2MB 충분) |
| 7B (14GB) | 0.06% | 29% (hot set OK) | 100% | 3–5% |
| 32B (64GB) | 0.01% | 6% | 100% | 5–10% |
| 70B INT4 (35GB) | 0.02% | 11% | 46% | 10%+ |

**4KB → 2MB 이득이 황금구간 (10–30%)**. 2MB → 1GB 는 작음 (3–10%).

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

**vLLM 코드 변경 불필요** (THP 는 커널 자동 promotion). 설정 후 vLLM **재시작** 만 필요.

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

장기 uptime 서버는 단편화로 **부분 확보 (12/40 등) 또는 0 확보**. 성공률 낮음.

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

### Phase 2 — 경로 B (1GB 런타임, 재부팅 없이) — **Phase 1 이득 확인 후**

- [ ] **1GB 페이지 확보 시도**: `echo 40 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages` + NUMA 노드별 분배. 실제 확보량 로그.
- [ ] **vLLM weight loader 수정**: `vllm/model_executor/model_loader/loader.py` 에 `HYBRID_HUGEPAGES=1gb` 분기. `torch.from_file` + 수동 `mmap(MAP_HUGETLB | MAP_HUGE_1GB)`.
- [ ] **KV cache allocator 수정**: `vllm/v1/worker/cpu_worker.py` 의 `VLLM_CPU_KVCACHE_SPACE` 참조부에 1GB mmap 경로 추가.
- [ ] **Fallback chain**: 1GB 실패 → 2MB (MAP_HUGE_2MB) → 4KB. `[HYBRID-HUGEPAGE] page_size=? weight_bytes=? kv_bytes=?` 로그 marker.
- [ ] **G0 sweep 재실행**: `measurement_results/<HW>/g0_03_1g/seqs<N>/` 로 분리 저장. Phase 1 (2MB) 대비 추가 이득 측정.

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
