# PLN_008 M0 — 기계 모델 v2 중간 상태 (2026-09-08 10:05)

## 구조 확정 (프로파일 분해 근거)
- 스텝 = Σ층 [GPU 비-expert + GPU hot-expert + CPU 노출]: **직렬**. 근거 = hot-80 (GPU 26 + CPU 대기 40 = 66) / hot-96 (커널 27.7 + 유휴 9.9 + 복사 1.9 = 39.5) 두 분해 모두 겹침 없음.
- hot-96+def4 C32 decode 스텝: 예측 38.7 vs 프로파일 39.0 (−0.8%) — GPU expert 항은 프로파일 실측 비례 자리표시 (격자 벤치로 교체 예정).

## TPOT 재예측 (prefill 항 미포함 상태): 중앙값 |오차| 24.9%, 전 케이스 음수
| 케이스 | 예측 | 실측 | 오차 | 해석 |
|---|---|---|---|---|
| hot-80 C16/C32/C64 | 43.3 / 63.6 / 83.3 | 53.3 / 83.3 / 134.0 | −19 / −24 / −38% | prefill 항 (≈12~17ms) 부재 + C64 는 추가 항 |
| hot-80 def2/def4 C32 | 54.8 / 51.3 | 80.1 / 78.9 | −32 / −35% | deferred 숨김 과대 (모델 12.3ms vs 실측 이득 3~4ms) |
| hot-96 C32 / +def4 C16 / C32 / C64 | 47.8 / 29.5 / 38.7 / 41.2 | 54.8 / 35.9 / 51.5 / 86.7 | −13 / −18 / −25 / −53% | prefill 항; C64 는 KV 압박 항 |

## 남은 항 (측정 예정, 순환 적합 금지)
1. prefill 토큰당 비용 — prefill 전용 벤치 (out=1, C=1/8/32) 진행 중
2. GPU hot-expert 커널 g(H, B, D_h) — 단독 격자 벤치 진행 중
3. deferred 가 cold 를 잡을 확률 — 라우팅 가중치 분포 기록 필요 (topk 가중치 순위별 cold 비율; 트레이스는 카운트만 있음)
4. KV 압박 항 — C64 셀의 retraction·재-prefill 증거 확인 후 정식화
5. hot-80 프로파일 사전 예측 (`PREREG_hot80_profile.md`): 스텝 63.5 / 유휴 33~38 / binding cpu_ddr

## KV 압박 증거 (hot-96+def4 C64, max-total-tokens 24576, 192 요청)
- 서버 로그 "KV cache pool is full. Retract requests" 20건, running-req 는 C=64 구간에서 ~45~47 (모델 B_eff = 24576/(512+64) ≈ 42.7 — prefix 100 은 radix 공유).
- 모델 반영: B_eff = min(C, KV_tokens/(고유 입력 + 출력/2)), retraction 재-prefill 비율 f_r = 0.5 × (1 − B_eff/C) **(가정, M1 에서 검증 대상)**.

## 12:30 상태 — 파라미터 4종 반영 후 재예측
- GPU expert 항: 격자 v2 (graph 재생) 최소자승 µs/층 = 39.7 + 4.26·D_h − 0.05·B (최대 오차 2.7%). decode 스텝 재예측 hot-96+def4: 40.0 vs 39.0 (+2.6%).
- prefill 항: 요청당 T_pf(b_p) 표 (322/71/19 ms @ b_p 1/8/32, 로그 보간) × B/out, b_p = C/4 **가정**.
- TPOT 재예측 9셀: 중앙값 |오차| 17.5%, 최대 40.2%. ±20% 밖 3셀 = hot-80 C16 (+20.4), hot-96 C32 (+22.1), hot-96+def4 C16 (+40.2), hot-96+def4 C64 (−29.1).
  - C16 과대: prefill 항 (b_p=4 → 19.3ms/스텝) 이 실측 (35.9−31.1 ≈ 4.8ms) 보다 4배 큼 → admission 배치 가정이 저동시성에서 틀림 (요청 완료가 동기화되어 b_p 가 C 에 가까움). 대안 규칙 검토: b_p = C/4 → b_p = max(C/2, 8)? — **새 셀로만 검증** (기존 셀로 규칙을 고르면 순환 적합).
  - C64 과소: KV 포화 thrash (retraction 20건, running 46) — 모델 범위 밖으로 표기, 정책은 회피.
- M0 게이트 (재예측 전부 ±20%) **미통과** 상태. 남은 항: deferred 적중 확률 (topk 덤프 진행 중), prefill admission 규칙, KV thrash.

## K1 위협 갱신 (분신 C, 12:20)
- 학술 사이트 (arxiv/usenix/acm/openreview/semanticscholar/alphaxiv/archive.org) 는 호스트·컨테이너 모두 차단 — huggingface.co 만 열림 (초록 API 가능). **전문 정독은 PDF 반입 없이는 불가** → 사용자에게 PDF 요청.
- MoE-Lens (2504.09345): CPU attention / GPU GEMM 오프라인 배치 하이브리드에서 "전체론적 성능 모델로 하드웨어 상한·달성 처리량 예측 (평균 94% 정확도)". D2 형 위협 — 무대 (온라인 서빙 TPOT vs 배치 처리량), 토폴로지 (CPU expert vs CPU attention), 결정 변수 (expert 캐시↔KV 분할) 가 다른지 전문 확인 필수. CoX-MoE (2605.17889): HF 미등재.

## M1 셀 선정 (12:45, `features/IDE_031/m1_cells.json`, 시드 20260908)
격자 H{64,80,96} × C{8,16,32,64} × ctx{640, 2000} × KV{gpu, hicache} = 48 − 기측정 6 = 42 후보. 4축 극단 (C=8 / C=64 / ctx=2000 / hicache) 각 1개 강제 후 무작위 → 12셀 (hicache 7, ctx2000 7, C=8 4, C=64 3, H=64 6). 예측은 M0 게이트 통과 후 `predictions.json` 으로 커밋. hicache 셀은 M2 의 HiCache×kt 생존 시험 (진행 중) 통과가 선행.

## deferred 항 (12:15, topk 덤프 300 층호출·62층, `eval/results/20260908_123000_pln008_m0_topk_dump/`)
- cold expert 가 토큰 가중치 하위 N 에 들 확률 p = 0.13 / 0.26 / 0.52 (N=1/2/4; hot-80·96 동일) = **N/8** — cold 는 가중치와 무관.
- 그런데 deferral 이득 실측은 3~4 ms/스텝 (모델의 cpu×N/8 = 9~12 ms 보다 훨씬 작음) → 숨김 창이 병목: 단일 CPU FIFO 큐에서 층 L 의 deferred 작업은 층 L+1 의 immediate 앞에 실행되므로 겹칠 수 있는 GPU 구간은 다음 층의 MoE 이전 부분 (attention + qkv/o GEMM + AR/2 ≈ 3.7 ms/스텝 @B32) 뿐. 모델: hidden = min(pre_moe_window, cpu × N/8). (기전 도출, 처리량 적합 아님.)

## 12:25 — 구조 불확실성 1건 (M1 부속 판별로 넘김)
- 층별 라우팅 통계 (`routing_stats_layer.json`): hot-96 B=32 에서 cold 0개 층 확률 평균 8.3% (hot-112: 36.5%) → "cold 없는 층 건너뜀" 효과는 hot-96 에선 미미 (층당 ~5µs). hot-96 셀의 +16~22% 잔차는 이것이 아님.
- 남은 가설: 층 안에서 CPU cold 작업과 GPU hot-expert 커널이 **부분 겹침** (submit host 노드가 expert 커널보다 앞서 실행) — hot-80 C32 TPOT 는 직렬 (−0.7%) 이 맞고, hot-96 decode 스텝은 직렬이 +16% 과대. 두 체제 사이에서 겹침 정도가 다를 수 있음 (CPU 병목 체제에선 큐 backlog 로 사실상 직렬, GPU 병목 체제에선 겹침 발생).
- **M1 부속 판별**: H ∈ {64, 80, 96} 각각 정상 상태 (45초 이후) 프로파일에서 유휴 vs (cpu 예측 − fused_moe 실측) 를 대조해 직렬/겹침 구조를 셀별로 판정. 모델은 직렬 유지 (현 재예측 중앙값 20.4%, 게이트 경계).
- SGLang `enable_unified_memory` 는 hybrid 모델 풀 통합 옵션 (UVM KV 아님) → 실행 중 KV 의 host 배치 수단 없음. M2 의 "DRAM KV" 는 HiCache 의 prefix 계층 의미로 재정의 (RESULTS in `*_m2_hicache_smoke/`).

## 12:30 — prefill admission 배치 실측 (스케줄러 거동, 서버 로그 "Prefill batch #new-seq")
- hot-96+def4 최종 검증 로그: running ≤16 구간 평균 3.4 (중앙값 1, p90 15) / 17~32 구간 평균 14.4 (중앙값 12) / >32 구간 평균 4.7. hot-80 C32 로그: 평균 9.0 (중앙값 7).
- 규칙 갱신: b_p = max(1, C/3). 남는 C16 잔차는 prefill 배치가 decode 와 겹치는 정도 (chunk 정책) 문제로 보이며 M1 셀 (C=8 ×4, C=16 ×1) 에서 검증.

## M0 게이트 상태 (12:35)
- 재예측 9셀 (TPOT): 중앙값 |오차| 18.3%, ±20% 밖 3셀 — hot-96+def4 C16 (+37%: 저동시성 prefill 겹침), hot-80 C64 (−21%), hot-96+def4 C64 (−25%: KV 포화 thrash). decode 스텝 (프로파일) 재예측: hot-96+def4 +16% (직렬 가정 과대 가능).
- **판정: M0 게이트 (전 셀 ±20%) 미통과.** 남은 잔차 2종은 기존 셀로 규칙을 고르면 순환 적합이 되므로 M1 의 새 셀 (C=8 ×4, C=64 ×3, 정상 상태 프로파일 H 3종) 에서 판별·수정한다 (PLN_006 규칙: 수정 1회, 새 셀로만 재검증). M1 예측은 현 모델 그대로 사전 등록.
- 파라미터 출처 요약 (전부 서빙 처리량 무관): CPU 층 법칙 (mb480) / DDR 간섭 (2-스트림 벤치) / 라우팅 층별 통계 (8-29 트레이스) / GPU expert 격자 (graph 재생) / prefill 요청당 비용 (out=1 벤치) / admission 배치 (스케줄러 로그) / deferred 적중 확률 (topk 덤프) / GPU 비-expert 범주 시간 (프로파일 커널 합).
predictions.json sha256 375f3be50472ec48 2026-09-08T12:26:01
