# SUB_252 — Layer-skip self-speculative decoding (R5, GPU-direct 신규) — no-go (2026-06-16)

> 10라운드 루프 R5. 신규성(=vLLM 미적용): vLLM spec은 별도 draft 모델·EAGLE·Medusa는 있어도
> **자기 모델 레이어-스킵 self-draft 없음**. GPU-direct(draft=부분 forward). 출력안전(rejection
> sampling=분포보존). 목표: skip된 cheap draft + full verify → decode 연산 절감.

## 타당성 probe (8B, 32층, teacher-forcing top1 일치 = accept 천장)
**꼬리 절단** (`exp/probe_layerskip.py`, logit-lens): skip2=0.537 / skip4=0.530 / skip8=0.354 / skip16=0.023.
**중간 블록 skip** (`exp/probe_midskip.py`, 실제 부분 forward):

| skip N (중간) | draft연산 c=(nl−N)/nl | accept a (top1) | 속도배율 1/(c+(1−a)) |
|---|---:|---:|---:|
| 2 | 0.94 | 0.815 | 0.89 |
| 4 | 0.88 | 0.739 | 0.88 |
| 8 | 0.75 | 0.322 | 0.70 |
| 12 | 0.62 | 0.228 | — |
| 16 | 0.50 | 0.131 | — |

중간 skip ≫ 꼬리절단(중간층 redundancy 통설 확인) 이나 **고accept는 소skip(연산절감 미미)에만**.

## 판정 = **no-go (구조적 net-negative)**
self-spec 속도배율 ≈ **1/[c+(1−a)]**. draft를 K번 순차 실행하므로 c(draft 연산비율)가 지배.
측정된 (c,a) 전 구간에서 **c+(1−a) > 1 → 배율 < 1 (무조건 느려짐)**. draft 연산비율과 accept가
커플링(적게 skip→비싸지만 정확 / 많이 skip→싸지만 부정확)되어 **training 없이는 sweet spot 부재**.
- vLLM이 이미 가진 해법(별도 1B draft c~1.4%·EAGLE 학습 head)은 이 커플링을 학습/별도모델로 깸.
- layer-skip이 win하려면 **layer-dropout 파인튜닝(LayerSkip 논문)** 필수 = serving 범위 밖.

## 함의 (R1~R5 메타)
- GPU-direct + 신규(vLLM 미적용) + **training-free** + net-positive = **사실상 공집합**.
  싼-draft×고-accept 또는 출력안전 연산절감은 전부 (a)비트폭[upstream] 또는 (b)학습된 draft[upstream]로 귀결.
산출물: `exp/probe_layerskip.py`, `exp/probe_midskip.py`.
