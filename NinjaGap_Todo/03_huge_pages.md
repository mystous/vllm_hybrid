# 03. Huge Pages 1GB

**Tier**: 0
**상태**: ⭕ 미구현
**예상 이득**: 5–15%

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

### 구현 경로

**옵션 A — explicit hugetlbfs mount**:
```bash
# 부팅 시 GRUB
hugepagesz=1G hugepages=40 default_hugepagesz=1G
# mount
mkdir -p /dev/hugepages-1G
mount -t hugetlbfs -o pagesize=1G none /dev/hugepages-1G
```
사용자 코드에서 `open()` + `mmap(MAP_HUGETLB | MAP_HUGE_1GB)`.

**옵션 B — THP (Transparent Huge Pages)**:
```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```
4KB → 2MB 자동 promotion. 1GB 는 지원 안 함 (Linux THP limitation). 간편하지만 1GB 미사용 시 효과 축소.

**옵션 C — `madvise(MADV_HUGEPAGE)`**:
allocate 후 hint. THP 모드에서만 의미.

### 현재 상태

- grep `MAP_HUGETLB` — 결과 없음
- grep `hugepagesz` — 결과 없음
- vLLM 내부 weight mmap 경로는 HuggingFace `safetensors` → `torch.load` → page fault on demand (4KB)
- IPEX oneDNN 은 내부 scratchpad 를 malloc 기반 (THP 범주)

### Container 환경 제약

- Docker/cgroup v2: `--cap-add=SYS_ADMIN` 또는 `--privileged` 필요 (hugetlbfs mount)
- Kubernetes: `hugepages-1Gi` resource + node kernel parameter 필요
- 재부팅 없이 runtime 에 1GB 페이지 확보는 **fragmentation 위험** — boot-time 할당 권장

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

- [ ] **컨테이너 feasibility 확인**: Docker run options, cgroup v2 설정, hugetlbfs mount 권한
- [ ] **dev 머신 선검증**: i9-12900KF (DDR5, 작은 TLB) 에서 1GB page 효과 확인 — weight 크기가 작아 효과 미미할 수 있음
- [ ] **GRUB 파라미터 설정**: `hugepagesz=1G hugepages=40` (40GB 확보, 7B BF16 full + KV cache 여유)
- [ ] **hugetlbfs mount point 설정**: `/dev/hugepages-1G`
- [ ] **vLLM weight loading path 수정**: `safetensors` → `mmap(MAP_HUGETLB | MAP_HUGE_1GB)` 로 교체 경로 조사
  - 후보 1: `vllm/model_executor/model_loader/loader.py` 의 `safetensors.safe_open` 주변
  - 후보 2: `torch.from_file` 직접 호출 + `mmap` pre-load
- [ ] **KV cache hugetlbfs 연결**: `VLLM_CPU_KVCACHE_SPACE` 가 참조하는 메모리 할당 경로
- [ ] **`MAP_HUGETLB` fallback**: 1GB 실패 시 2MB, 2MB 실패 시 4KB. `[HYBRID-HUGEPAGE]` 로그 marker 추가
- [ ] **실측**: Huge Pages on/off 비교 (동일 model × workload), `perf stat -e dTLB-load-misses` 차이 확인

---

## 성공 조건

- `perf stat -e dTLB-load-misses,dTLB-load-miss-walk-pending` 기준 **dTLB miss 80% 이상 감소**
- decode per-step 시간 5–15% 감소 (G0 baseline 대비)
- PPL 변화 없음 (수치 계산 무관)

---

## 의존성

- **선행**: §01 G0 계측 (baseline 확보)
- **병행**: §04 IPEX WoQ INT8, §05 OMP env
- **후속**: 모든 기법 (항상 깔려 있어야 하는 infra)

---

## 리스크

- **컨테이너 권한 불허**: 운영팀 승인 필요. dev 머신으로 선검증 후 승인 요청
- **1GB 페이지 fragmentation**: boot-time 할당 필수
- **HuggingFace/safetensors 라이브러리가 mmap path 에 hardcoded**: PyTorch `torch.from_file` + 수동 mmap 경로 구축 필요 — non-trivial
- **이득 측정 어려움**: 다른 기법 (WoQ INT8, ISA dispatch) 과 동시 적용 시 단독 기여도 분리 필요

---

## 스택 호환성

**모든 후속 기법과 독립**. 항상 깔려 있어야 함. Huge Pages 로 확보한 TLB 여유는 §15 AMX pre-pack, §13 T-MAC LUT 등 memory-bound 기법의 이득 상한을 높임.

---

## 관련 코드 위치

- `vllm/model_executor/model_loader/loader.py` — weight loading
- `vllm/v1/worker/cpu_worker.py` — `VLLM_CPU_KVCACHE_SPACE` 참조
- `vllm/v1/core/kv_cache_manager.py` — KV allocation
- `eval/envs/h100x8_*.env` — `VLLM_CPU_KVCACHE_SPACE` 설정
