# SUB_258 — DRAM KV-offload: CPU/DRAM tier 가 GPU-only 를 이기는 regime 측정

*측정 2026-06-17, 1×B200(7B), 모든 수치 본 세션 실측. CERES/harvest 명제 검증.*

## 질문 (프로젝트 코어)
"기존 (논문이 다룬) 분야를 **CPU/서버 하드웨어를 극대화**해서 GPU-only 와 동등 또는 그 이상으로
만들 수 있는가?" — KV-cache 가 GPU 에 다 안 들어가는 regime 을 대상으로 직접 A/B.

## 설정
- 모델 Qwen2.5-7B-Instruct, TP1, `--enable-prefix-caching`, `--max-model-len 8192`.
- GPU KV cache 를 `--num-gpu-blocks-override` 로 인위 축소(자연값 157,899 blk → 강제 축소).
- **워크로드**: distinct 긴 prefix 24개(각 ~3,050 tok) × 4 round 재사용, suffix 만 다르게,
  max_tokens=32, concurrency 8 → 96 req. working-set(prefix KV) ≈ **73,000 tok ≈ 4,560 blk**.
- **A (GPU-only)**: offload off. evict 된 prefix 는 재사용 시 prefill **재계산**.
- **B (GPU+DRAM)**: `--kv-offloading-size 60 --kv-offloading-backend native`
  (`--disable-hybrid-kv-cache-manager` 필요). evict 블록이 DRAM 으로 내려갔다가 재사용 때 **fetch**.
  전송 = pinned host mem(cudaHostRegister) + async H2D/D2H DMA (PCIe). **DSA(/dev/dsa) 미사용** —
  vLLM native backend 은 copy engine 만 씀(정직: DSA 가속은 미구현, 향후 여지).

## 결과 1 — 고정 cache(24K tok), 3-pass paired (변동 <1%)
| 구성 | out_tps | TTFT mean | TTFT p90 | wall |
|---|---:|---:|---:|---:|
| A GPU-only | ~932 | 0.114 s | 0.160 s | 3.27 s |
| **B GPU+DRAM** | **~1650** | **0.036 s** | 0.049 s | 1.85 s |
| **이득** | **+76.7%** | **−68%** | −69% | −43% |
출력 토큰 수 A 3051 ≈ B 3042 (greedy 동등; temp0 생성길이 미세차).

## 결과 2 — 크로스오버 특성화 (GPU cache 크기 sweep)
working-set ≈ 73K tok 기준, GPU cache 를 바꿔가며:
| GPU cache | ws/cache | A tps | B tps | **B/A** | regime |
|---:|---|---:|---:|---:|---|
| 12.8K tok | ≫1 | 920 | 1537 | **+67%** | heavy evict → DRAM 회수 |
| 24K tok | ≫1 | 942 | 1660 | **+76%** | evict → DRAM 회수 |
| 48K tok | ~1 | 1790 | 1610 | **−10%** | GPU 충분 → offload 오버헤드 |
| 112K tok | <1 | 1775 | 1783 | **~0%** | 완전적재 → evict 0, offload 무동작 |
(48K↔112K 사이 4.8K-blk 점은 B boot 일시 실패로 미측정; 경계는 48K/112K 로 bracketed.)

## 메커니즘 검증 (B `/metrics`, 24K cache)
- **External prefix cache hit rate 80–86%** (DRAM tier 가 재사용 대부분 서빙).
- GPU prefix cache hit ~1.5% (GPU cache 작아 evict).
- CPU→GPU fetch 15.87 GB / 0.29 s ≈ **54 GB/s** (PCIe gen5 pinned DMA). GPU→CPU 2.55 GB / 0.049 s.
- 즉 "fetch(≈ms) ≪ recompute prefill(≈수십 ms)" 가 win 의 물리적 원인.

## 판정
- **YES**: KV-pressured regime(working-set > GPU cache)에서 idle DRAM tier 를 켜면 GPU-only 대비
  **+67~77% throughput / −68% TTFT**. 출력 동등(greedy 동일). = harvest 명제 실증.
- 단 GPU 에 여유가 있으면(cache ≥ working-set) **무이득~−10%**(offload 순오버헤드).
  → harvest 는 **상황 적응형으로 켜야** 손해 안 봄.

## 신규성 (정직)
- **메커니즘 자체는 신규 아님**: vLLM `native` CPU KV offload + OffloadingConnector 기존 기능
  (LMCache/Mooncake 동류). 본 작업은 **실증 + 크로스오버 정량화**이지 신규 알고리즘 아님.
- **신규 여지(미구현)**: (a) **DSA-가속 전송** — 현 native 는 cudaMemcpy(copy-engine/PCIe).
  /dev/dsa ×2 로 D2H/H2D 를 SM/copy-engine 비점유로 내리면 decode 와 진짜 overlap (CERES 본령).
  (b) **적응 게이트** — KV 압력 측정해 offload on/off 자동 전환(48K 의 −10% 회피).

## 결과 3 — 70B 스케일링 (win이 더 커짐) + fetch-vs-recompute 헤드룸 검증
Llama-3.1-70B-NVFP4(40GB) 단일 B200, KV cache 40K tok(자연 22,347blk→2,500blk override).
70B KV = **320 KB/tok**(7B의 5.7×) → fetch 부피 큼.
| regime | A GPU-only | B GPU+DRAM | B/A |
|---|---:|---:|---:|
| 중압(working-set 60K, ws/cache 1.5×) | 408 tps / TTFT 0.055 | 405 tps / 0.059 | ~0% (대부분 GPU hit, fetch≈0) |
| **고압(working-set 180K, ws/cache 4.5×)** | **81 tps / TTFT 0.337** | **160 tps / 0.097** | **+97.5% / TTFT −71%** |
- 고압서 실제 fetch 발생: CPU→GPU **97.6 GB / 1.76s = 55.4 GB/s**, GPU→CPU 33.7GB/0.61s=55GB/s.
- **harvest win은 70B서 더 커짐(+97.5% > 7B +77%)** — 모델 클수록 recompute-prefill 비용↑이라 fetch 회피가치↑.

**후보 (c) fetch-vs-recompute 공동스케줄링 = ❌ 헤드룸 0 (실측)**:
- vLLM엔 `recompute_kv_load_failures`(fetch 실패시 recompute fallback)만 존재, 능동 co-scheduling 無 = 기술적 신규.
- **그러나** B의 병목 = **decode(wall 9.0s), fetch 아님(1.76s, async 스트림 오버랩)**. fetch는 이미 critical-path
  밖 → fetch↔recompute 재배분으로 9s decode floor 못 움직임. 게다가 fetch(TTFT 0.097)가 recompute
  (TTFT 0.337)보다 3.5× 싸서 "싼 자원(fetch)→비싼 자원(recompute) 이전"은 손해. **arbitration 이득 ≈ 0, 死.**

## 결론 종합 (신규성)
SR-006 harvest win은 7B +77% / **70B +97.5%** 로 견고·확장. 단 3개 신규 후보 전부 막힘:
(a) DSA=아키텍처 dead-end(GPU↔host PCIe 못닿음), (b) 적응게이트=신규성 약함, (c) fetch-vs-recompute=
decode-bound라 헤드룸 0. → **메커니즘 신규성 없음. SR-006의 가치 = harvest 명제의 강력한 실증/특성화
(CERES 논문 evidence)이지 신규 알고리즘 아님.** [[largemodel-b200-comm-bound-characterization]] 패턴 재확인.

## 산출물
- `bench_kv.py`(워크로드 클라이언트), `serve.sh`(A/B 서빙), `sweep_blocks.sh`(크로스오버).
- `runs/results.jsonl`(고정 cache paired), `runs/sweep_blocks.jsonl`(sweep), `runs/serve_*.log`.
