**↑ 부모**: [`README`](README.md) (IDE_006) · **↟ 조부**: [`shadow_assists/README.md`](../../README.md)

---

# IDE_006 — 4 차 재정의 검토 · 결정 history (NEO 식 Asymmetric GPU/CPU Pipelining)

| 항목 | 값 |
|---|---|
| 결정 일자 | 2026-04-29 |
| 결정 단계 | 검토 + 결정 완료 (코드 미반영 — 별도 branch `feat/ide006-neo-asymmetric` 에서 적재 예정) |
| 영향 범위 | IDE_006 의 *시그니처 메커니즘* 전환 — "같은 request 안 hot/cold split + LSE merge" → "request 단위 GPU/CPU exclusive ownership + asymmetric pipelining" |
| 출발 branch | `main` (a30d90dddb) — IDE_006 의 초기 상태 코드 |
| 문서 import | `feat/ide006-cold-kv-cpu-partial-attention` (1f6fb8a7ec) 의 IDE_006 디렉토리 28 개 .md + id_registry / shadow_assists/README |
| 코드 적용 | 미반영 — 다음 turn 에서 NEO 식 신규 path 설계 + 재사용 코드 (TSK_001/003/004/010) cherry-pick |

---

## 1. 검토 history — 어떻게 NEO 4 차 재정의에 도달했는가

### 1.1 · TSK_011 sweep 결과 + 단일 단계화 (2026-04-28 ~ 2026-04-29)

`TSK_011` prod sweep (2026-04-28) 의 lp ~3.43 발산이 *fallback path / partition path 모두 같은 cold KV source (CPU page cache)* 사용으로 root cause 입증. fallback 만으로 D-ii 봉합 불가능 → `TSK_012` (decode-time cold reload + 진짜 evict) 분리 발급.

`TSK_009` fix v4 prod 검증 (`eval/results/20260429_043734_*_tsk009_validation/`) 으로 invariant 1 (속도) 1.078~1.103× / invariant 2 (CPU 활용) 0.00% 입증. fundamental Q dependency 로 layer-안 partition path 만으로는 향상 영역 없음.

`TSK_005` (cross-layer pipeline) 가 invariant 2 의 유일한 향상 영역으로 식별 → plug-in 형식 설계 후 사용자 지적 — `Q dependency + GPU 가 진짜 Q 가지면 CPU 결과 무용` dilemma 발견 → **TSK_005 기각**. CPU partial 의 진짜 가치 영역 = cold blocks 가 진짜 GPU evict 되는 시점의 reload 대체 → `TSK_012` 영역.

`TSK_012` 본문이 Phase 1 (D-ii 봉합) / Phase 2 (race) 단계로 재작성 → 사용자 지적 *"Phase2 는 cold tier 수정 안한 바닐라 버전보다 느려지는거 아냐? 이미 테스트 결과들이 말해주는거잖아"* → Phase 2 race 제거 (PCIe Gen4 32 KB cold block 0.002 ms vs CPU partial 6.4 ms 의 3000× 격차로 race 의 CPU win 영역 fundamental 작음). **TSK_012 단일 단계화** (commit `1f6fb8a7ec`).

### 1.2 · 사용자 지적 — mirror 가 더 빠르지 않은가 (2026-04-29)

TSK_012 단일 단계화 후 사용자 지적:

> "mirror를 가지고 있는게 개선한 것 보다 빠른거 아냐?"

**정확한 지적**. KV pool 충분 영역에서 vanilla mirror (cold blocks 가 GPU 잔류) 가 TSK_012 의 진짜 evict + reload 보다 *항상 빠름* — reload PCIe 비용이 추가되기 때문. TSK_012 의 가치 영역 자체가 흔들림.

추가 의문: mirror 가 *진짜* mirror 라면 cold-tier ON, IDE_006 OFF 회차가 baseline 과 lp 3.428 발산할 이유가 없음. 실제로는 발산 → mirror 가 *진짜* mirror 인지 / store path 에 정밀도 손실 / cold blocks read path 가 GPU 사본 vs CPU 사본 사용 여부 등 vLLM 코드 dive 필요 영역.

### 1.3 · NEO 논문 검토 — workload 우위 영역 식별 (2026-04-29)

사용자 지시:

> "Workload가 cold tier를 CPU로 내려서 더 좋은 경우가 있을 것 같아. NEO 논문을 보면 나올 것 같아."

NEO 논문 ([arXiv 2411.01142](https://arxiv.org/abs/2411.01142), [MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html), [GitHub](https://github.com/NEO-MLSys25/NEO)) 검색 결과 정량 우위 영역 정의:

| GPU | GPU memory | NEO 의 throughput gain |
|---|---|---|
| T4 | 16 GB | up to **7.5×** (output sweep 영역에서 **750%**) |
| A10G | 24 GB | up to 26% (강한 CPU 시 **79.3%**) |
| H100 | 80 GB | up to 14% |

→ **GPU memory 가 작을수록 우위가 압도적**. 사용자 가설 (workload 별 우위 영역 존재) 의 정량 입증.

### 1.4 · NEO 의 fundamental 메커니즘 = request 단위 분할 + asymmetric pipelining

NEO 의 핵심 (검색 결과로 확인):

> "The KV cache system of NEO is divided into two separate components: the 'GPU-cache' located in the GPU's HBM, and the 'CPU-cache' located in the CPU main memory. For any request that has already been prefilled in the system, its KV cache will either reside entirely in the GPU-cache—designated as a 'GPU-request'—or entirely in the CPU-cache."

> "Asymmetric pipelining runs two asymmetric sub-batches concurrently: one offloads the decoding attention computation and KV cache of a subset of requests into the CPU, and another one runs the rest in the GPU."

> "To achieve full GPU-CPU overlapping, Neo integrates the prefilling stage computation into the GPU decoding sub-batch, so that the prefilling stage computation (in GPU) also happens in parallel with the CPU attention computation."

핵심: **분리 단위 = "request 통째로"**. 같은 request 의 KV 가 GPU/CPU 양쪽에 *섞이지 않음*. mirror 도 split 도 아닌 *exclusive ownership*.

### 1.5 · Q dependency dilemma 회피 — *fundamental 회피*

사용자 질문:

> "그럼 NEO는 Req가 CPU와 GPU에 완전히 나뉘어서 처리 된다는거야? 아니면 Q값만 CPU에서 처리 해서 GPU에 던져 준다는거야?"

정확한 NEO 의 동작 (decode 한 step 안 layer 별 흐름):

```
─── Layer N (CPU-request) ─────────────────────────────┐
  ① QKV linear projection   ← GPU 가 함 (Q, K, V 모두 GPU 가 생성)
  ② Q 만 GPU → CPU 전송      ← 작은 데이터 (BS × head_dim)
  ③ Attention(Q, K_cpu, V_cpu) ← CPU 가 함 (KV 는 CPU 에 *상주*, reload 없음)
  ④ Attention 결과 CPU → GPU ← 작은 데이터
  ⑤ Out projection / FFN    ← GPU 가 함
─────────────────────────────────────────────────────────┘
```

| 영역 | IDE_006 / TSK_005 (기각) | NEO |
|---|---|---|
| 분리 단위 | 같은 request 의 hot/cold 부분 분할 | **다른 request 통째로** |
| Q 의존 | layer N+1 의 진짜 Q 는 GPU 가 가짐 → CPU 결과 무용 | **CPU 가 그 request 의 Q 를 직접 사용** → 의존 없음 |
| 동기화 | layer 단위 GPU↔CPU 동기 필요 | request 단위 결과만 합치면 됨 (decoding 한 step 끝났을 때만) |
| dilemma | fundamental | **만나지 않음** |

NEO 가 dilemma 를 회피하는 *진짜* 메커니즘:

1. **KV 가 CPU 에 *상주*** — IDE_006 의 cold tier 처럼 reload 가 *전혀 없음*. GPU pool 에 그 request 의 KV 자체가 *없음*.
2. **GPU 는 그 request 에 대해 attention 을 *안 함*** — IDE_006 / TSK_005 의 dilemma 였던 "GPU 가 진짜 Q 가지면 paged FA full 가능 → CPU 결과 무용" 이 NEO 에선 발생 안 함. GPU 는 *다른 request 의 prefilling 으로 바쁨* → 그 request 의 attention 을 *처리할 의지조차 없음*.

```
같은 시점:
  GPU: ┌─ GPU-request A 의 decoding attention 처리 ─┐
       └─ CPU-request B 의 prefilling 처리 (다른 단계!) ─┘
  CPU:  ┌─ CPU-request B 의 decoding attention 처리 ─┐ (asymmetric overlap)
```

CPU 가 B 의 attention 진행 동안 GPU 는 *그 request 를 못 도와줌* (다른 일 중) → CPU 결과가 *유일한* 결과 → CPU 결과 무용 안 됨.

---

## 2. 기존 8 개 적용 TSK 의 NEO 식 적용 시 운명

기호: ✅ 그대로 사용 · 🔶 부분 사용 · ❌ 미사용

| TSK | 상태 | 적용된 내용 | NEO 적용 시 |
|---|---|---|:-:|
| TSK_001 | 완료 | LSE-반환 CPU partial-attention kernel (Python ref + portable C++ + wrapper) | 🔶 |
| TSK_002 | 활성 (§4.2~§4.6 적용) | scheduler / attention metadata 의 hot/cold partition 통합, Q D2H stream 분리, reload sync (§4.5c) | 🔶 |
| TSK_003 | 활성 | AVX-512 + AMX prod SIMD kernels | ✅ |
| TSK_004 | 활성 | NUMA-aware (worker bind + thread affinity + worker 별 cpulist) | ✅ |
| TSK_007 | 완료 | (코드 변경 없음 — 옵션 A 결정) | ✅ |
| TSK_009 | 활성 (fix v4 적용) | `hot_cold_attention` 의 non-blocking poll + done 분기 helper + paged FA full inplace | ❌ |
| TSK_010 | 활성 (단계1~2 적용, default off) | `forward_partial_with_lse_sub_batched` + per-thread `omp_set_num_threads` 인프라 | ✅ |
| TSK_011 | 활성 (fallback path 적용) | `_resolve_cold_deadline_s` / `_record_cold_fallback_breadcrumb` / `_fallback_full_fa_paged` + sweep wrapper | ❌ |

### 2.1 · 부분 사용 (🔶) 의 세부

| TSK | 사용 ✅ | 미사용 ❌ |
|---|---|---|
| TSK_001 | kernel 코어 (Q·Kᵀ softmax + V) | LSE 반환 부분 (NEO 는 partial 합산 안 함) |
| TSK_002 | Q D2H stream 분리 인프라 (NEO 의 GPU→CPU Q transfer 에 재활용) | hot/cold partition 통합 + scheduler / metadata 분리 + reload sync |

### 2.2 · 요약

- ✅ **그대로 사용**: 4 개 (TSK_003 / 004 / 007 / 010) — kernel 가속 + 자원 인프라
- 🔶 **부분 사용**: 2 개 (TSK_001 / 002) — kernel 코어와 transfer 인프라만
- ❌ **미사용**: 2 개 (TSK_009 / 011) — hot/cold split 의 layer-안 협력 코드

### 2.3 · 대기 / 기각 TSK

| TSK | 상태 | NEO 적용 시 |
|---|---|---|
| TSK_005 | 기각 (Q dependency dilemma) | NEO 에서 dilemma 자체가 회피되므로 reference 의의는 보존, 본 ID 는 그대로 기각 |
| TSK_006 | 대기 (Q chunk pipelining) | NEO 와 직교 — 폐기 |
| TSK_008 | 대기 (hot/cold 분할 정책) | NEO 는 분할 자체 안 함 — 폐기 |
| TSK_012 | 대기 (decode reload + 진짜 evict) | NEO 식 *exclusive ownership* 으로 본문 재작성 또는 폐기 후 신규 TSK 발급 결정 영역 |

---

## 3. branch 분리 결정 — 옵션 A 채택

### 3.1 · 두 옵션 비교

| 차원 | 옵션 A (main 에서 새 branch + 기존 코드 cherry-pick) | 옵션 B (현재 branch 를 NEO 식으로 변경) |
|---|:-:|:-:|
| 깨끗한 NEO 식 측정 | ✅ | ❌ (잔존물 noise) |
| 두 architecture 분리 보존 | ✅ | ❌ (한 branch 에 hot/cold split + NEO 섞임) |
| 실패 시 회귀 비용 | 작음 (branch archive 후 main 복귀) | 큼 (dead 제거 비가역) |
| 재사용 코드 가져오는 비용 | cherry-pick 5 개 (작음) | 0 |
| git history 가독성 | ✅ | ❌ |

### 3.2 · 채택 사유

옵션 B 의 유일한 장점 (cherry-pick 불필요) 보다 옵션 A 의 4 가지 장점 (history 분리 / 회귀 단순 / back-out 가능 / 측정 순수성) 이 압도적으로 큼. 특히:

1. **측정 순수성이 결정적**. NEO 식 architecture 가 정말 우위 영역을 가져오는지 *측정으로 입증* 해야 진행 의의가 있음. 옵션 B 는 TSK_002 잔존 인프라가 측정에 들어가 NEO 의 순수 가치 영역을 흐림.
2. **cherry-pick 비용 자체가 작음**. TSK_001 (kernel) / 003 (SIMD) / 004 (NUMA) / 010 (sub-batching) 모두 vLLM core 영역 변경 적은 부분이고 self-contained — 표준 cherry-pick 작업.
3. **실패 가능성 대비**. NEO paper 의 정량 결과 (T4 7.5× / H100 14%) 를 IDE_006 환경 (Llama-70B + TP=8 + H100×8) 에서 재현 가능한지는 *측정 후에야 확정*. 옵션 A 는 가치 영역 못 나오면 branch archive 후 main 그대로, 옵션 B 는 dead 코드 제거가 *이미 일어난 상태* 라 revert 부담 큼.
4. **IDE_006 의 4 차 재정의** 라는 framing 변경이 새 branch 와 함께 하는 게 문서·코드 정합 측면에서 자연스러움.

### 3.3 · 새 branch

- 이름: `feat/ide006-neo-asymmetric` — NEO 식 *asymmetric pipelining* 의 핵심 메커니즘을 직접 표현
- 출발 commit: main `a30d90dddb` (2026-04-29 기준)
- 첫 commit: 본 IDE_006 문서 디렉토리 28 개 .md + id_registry / shadow_assists/README 의 *문서 import* (commit `d2ae1c2a61`)
- 두 번째 commit: 본 NEO_redesign.md 신규 + IDE_006 README 4 차 재정의 entry 갱신

### 3.4 · 기존 branch (`feat/ide006-cold-kv-cpu-partial-attention`) 의 처분

- **archive 보존** — git history 는 그대로 보존, push 된 origin 도 그대로
- 필요 시 cherry-pick 으로 재사용 가능 코드 (TSK_001 kernel / TSK_003 SIMD / TSK_004 NUMA / TSK_010 sub-batching) 를 새 branch 로 가져옴
- NEO 식이 가치 영역 못 가져오는 것으로 측정 결과 입증되면 → 기존 branch 가 archive 상태에서 *부활* 가능

---

## 4. NEO 식 architecture 의 신규 TSK 영역 (다음 단계)

NEO 의 메커니즘을 vLLM 위에 적재하기 위한 신규 TSK 후보 (실제 발급은 다음 turn 에서 사용자 결정 후):

| 신규 TSK 후보 | 역할 |
|---|---|
| Request-level scheduler | prefilling waitqueue / GPU decoding runqueue / CPU decoding runqueue 의 vLLM scheduler 통합 |
| KV cache exclusive ownership | vLLM `OffloadingConnector` 의 mirror 정책을 *request 단위 GPU/CPU exclusive* 로 변경. 이전 `TSK_012` 본문 재정의 또는 폐기 후 신규 발급 |
| Asymmetric pipelining | GPU sub-batch (decoding GPU-requests + prefilling all) + CPU sub-batch (decoding CPU-requests) 두 sub-batch 동시 실행 hook |
| Load-aware scheduling | request 의 GPU/CPU 배정 결정 heuristic — NEO §4 영역 |

(TSK_001 kernel + TSK_003 SIMD + TSK_004 NUMA + TSK_010 sub-batching 은 위 TSK 들의 CPU sub-batch attention 구현에 그대로 사용)

---

## 5. NEO 식 적용 전 Phase 0 — 코드 dive 가 우선

NEO paper 본문 (특히 §3 system design + §4 scheduling heuristic + §5 evaluation) 의 직접 확인이 *진입 전 필수* 영역:

| 질문 | 결정 영향 |
|---|---|
| CPU 가 attention 을 어떻게 효율적으로 처리하는가 (AVX-512 / AMX 활용?) | TSK_001 kernel 의 cherry-pick 완전 가능성 |
| Load-aware scheduling 의 결정 heuristic | 신규 TSK Load-aware scheduling 의 본문 |
| vLLM 의 mirror 정책의 KV pool overflow 동작 (preemption / reject / partial-evict) | NEO 식 exclusive ownership 의 vLLM 통합 영역 |
| asymmetric pipelining 의 두 sub-batch 동시 실행이 vLLM 의 batching loop 에 들어갈 hook 위치 | 신규 TSK Asymmetric pipelining 의 본문 |
| NEO GitHub repo 의 vLLM 위 patch 영역 — 어느 vLLM 파일들이 수정되었는가 | 모든 신규 TSK 의 변경 범위 정합 |

본 환경에서 arxiv.org 가 timeout 으로 fetch 실패. 사용자 환경에서 직접 PDF 다운로드 후 본 디렉토리에 두기 (예: `super_power/papers/`) 또는 GitHub repo (`NEO-MLSys25/NEO`) clone 후 코드로 직접 확인 — 두 옵션 중 사용자 결정.

---

## 6. References

### NEO 논문
- [arXiv 2411.01142](https://arxiv.org/abs/2411.01142)
- [MLSys 2025 proceedings](https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html)
- [Harvard 사본 PDF](https://yangzhou1997.github.io/paper/neo_mlsys25.pdf)
- [GitHub repo (MLSys 25)](https://github.com/NEO-MLSys25/NEO)
- [MLSys 2025 slides](https://mlsys.org/media/mlsys-2025/Slides/3230.pdf)

### IDE_006 내부 history
- [`README`](README.md) — IDE_006 spec (1/2/3 차 재정의 본문)
- [`TSK_009`](TSK_009.md) — fix v4 prod 검증 결과 (invariant 1 / invariant 2 측정 출처)
- [`TSK_011`](TSK_011.md) — sweep 결과 (lp ~3.43 발산 입증)
- [`TSK_012`](TSK_012.md) — 단일 단계화 본문 (본 4 차 재정의의 직전 framing)
- [`TSK_005`](TSK_005.md) — Q dependency dilemma 기각

### 측정 결과 출처
- TSK_011 sweep — `eval/results/20260428_041131_*` (deadline=100ms) / `..._042424_*` (deadline=1000ms) / `..._025616_*` (비활성)
- TSK_009 fix v4 validation — `eval/results/20260429_043734_*_tsk009_validation/`
- **NEO 4 차 재정의 vanilla baseline** (정식 논문 reference) — [`PLN_001_neo_baseline_results.md`](PLN_001_neo_baseline_results.md) — Llama-3.3-70B + TP=8 + 5000 × 50:50 (4.7 시간) 의 KV 한계 영역 throughput 1,609 prompt_tps + concurrent 134/256 = 52% 가 NEO 의 *진짜 가치 영역* 발현 baseline

---

## 7. Change Log

| 날짜 | 변경 | 사유 |
|---|---|---|
| 2026-04-29 | NEO_redesign.md 신규 발행 (본 문서) | 사용자 결정 (2026-04-29) — TSK_012 단일 단계화 후 mirror vs evict 의문 + NEO 논문 검토 + request 단위 분할 + asymmetric pipelining 메커니즘 발견 → IDE_006 의 4 차 재정의 결정. 본 문서가 검토 + 결정 history 의 단일 출처. 새 branch `feat/ide006-neo-asymmetric` (main `a30d90dddb` 에서 fork) 의 두 번째 commit 으로 적재. NEO 식 architecture 의 코드 적재는 본 문서 + IDE_006 README 갱신 후 별도 commit 들로 진행. |

---

**↑ 부모**: [`README`](README.md) (IDE_006) · **↟ 조부**: [`shadow_assists/README.md`](../../README.md)
