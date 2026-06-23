# SUB_249 — 롱컨텍스트 KV 저랭크 압축 (란초스 본령) — 전제 검증 (2026-06-16)

> **전제 PASS: KV(특히 V)는 실제로 저랭크.** FP4 양자화 오차(high-rank 노이즈, TSK_049)와 정반대 →
> 란초스/저랭크가 닿는 영역 확인. 단 압축비 보통(K ~1.8×, V 1.8~5.7×), 구현 heavy·기존연구 중복.

## 측정 (Llama-3.1-8B, seq=3000, K/V SVD 실효 rank @99% 에너지, head_dim=128)
| | rank99/128 | 압축비 |
|---|---:|---:|
| K (L0/16/31) | 69~77 | ~1.8× |
| V (L0) | 22.5 | 5.7× |
| V (L16) | 71.8 | 1.8× |
| V (L31) | 42.5 | 3.0× |

→ **V가 K보다·초기레이어가 후기보다 더 저랭크.** rank99 ≪ 128 (특히 V) = 저랭크 실재.

## 판정 / 다음
- **전제 성립** (FP4와 달리). 란초스 top-k(matvec)로 KV 저랭크 추출 가능.
- 단 **이득 보통**(99% 에너지 기준 1.8~3×), 90% 기준이면 더 크나 정확도 게이트 필요.
- **이득 레짐**: 롱컨텍스트/대배치(어텐션 KV-대역폭-bound). 짧은컨텍스트 GPU-compute-bound엔 무관.
- **리스크**: 구현 heavy(커스텀 저랭크 어텐션 커널), 기존연구(Eigen Attention/Loki/ASVD-KV) 중복 — 신규성은 별도 검토.
산출물: `exp/kv_rank_probe.py`.

## PoC + 신규성 검토 (2026-06-16) — 전제 O, 신규성 X

**PoC (rank-r KV 절단 어텐션 출력 오차, 8B):**
| rank | 압축 | relerr |
|---|---:|---:|
| 64 | 2× | 0.118 |
| 32 | 4× | 0.379 |
| 16 | 8× | 0.617 |
→ **naive per-head SVD = 2×에 12% 오차** (타이트 게이트 부족, 4×는 붕괴). 문헌의 "M-LRD 정확도
저하" 그대로. 경쟁력 있으려면 group-head+activation-aware+캘리브(Palu/Eigen) 필요.

**신규성 검토**: KV 저랭크는 **포화 연구영역** — Eigen Attention(60%압축), Palu(ICLR'25, 50%/1.89×),
xKV(cross-layer SVD), KQ-SVD(provable), ASVD, LoRC, 그리고 **vLLM에 MLA(latent KV) 백엔드 이미 존재**.
naive 란초스로는 이들에 못 미치고, 매칭하려면 기존연구 재구현.

**종합 판정**: 전제(KV 저랭크)는 성립하나 — (a) naive 란초스 SVD 약함(2×/12%), (b) 포화 연구영역+
vLLM MLA 기존 → **신규 기여 어려움.** 세션 전체 패턴 재확인: 통하는 길은 이미 문헌/upstream이 점유.
산출물: `exp/kv_rank_probe.py`, `exp/kv_lowrank_poc.py`.
