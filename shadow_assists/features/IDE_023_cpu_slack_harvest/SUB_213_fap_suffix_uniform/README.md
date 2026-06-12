# SUB_213 — SUB_212 Confounder 재검증 (FaP 가설) + FaP×suffix 양립 (Uniform Draft Padding)

> **status**: 활성 (2026-06-11 신설)
> **parent**: `IDE_023` (CPU Slack Harvesting) / `TSK_043`
> **predecessor**: `SUB_212` (6-point coverage), `SUB_210`/`SUB_211` (FaP +30.1% B3 lever)

## 1. 발견 — SUB_212 "host DSA confounder" 결론의 오류

SUB_212 는 TSK_042 vanilla(8,850 tps, Llama-8B mix) 대비 신규 sweep vanilla(12,089)
의 +36% 차이를 **호스트 DSA WQ enable (2026-06-08 mtime)** 로 귀속시켰다.
본 SUB 의 재조사가 이를 반박:

| # | 증거 | 의미 |
|---|---|---|
| 1 | TSK_042 boot log: `cudagraph_mode=PIECEWISE` | 06-02 baseline 은 FaP 아님 |
| 2 | SUB_212 전 boot log: `cudagraph_mode=FULL_AND_PIECEWISE` | `sweep_corpus.sh:120` 이 명시 전달 (주석 "TSK_042 setting" 은 **오기**) |
| 3 | `/sys/bus/dsa/devices/wq0.0/clients = 0` | vanilla 측정 중 **어떤 프로세스도 WQ 미사용** — DSA 데이터 이동 자체가 없었음 |
| 4 | `/etc/ld.so.preload` 부재, DTO/DML 미설치 | 투명 memcpy→DSA 오프로드 경로 부재 |
| 5 | uptime 16.1일 (부팅 05-26) | 06-08 WQ enable 은 런타임 작업, 리부트 confounder 배제 |
| 6 | WQ 이름 = `lhc0` | WQ 는 LHC phase 작업이 만든 것 (시점 우연 일치) |
| 7 | SUB_210/211: FaP **+30.1%** (Llama-70B) 기측정 | +36% 와 같은 자릿수 — FaP 가설과 정합 |

**시그니처 재해석**: vanilla 는 균일 decode batch → FULL graph 적중 → +33~+36%.
suffix 는 draft 길이 가변 → uniform-decode 조건 (`q_len == 1+K` ∀req) 미충족 +
K=32 면 32req×33tok=1,056 > `max_cudagraph_capture_size=512` → **FULL graph 0% 적중**
→ FaP 무효과 (−5~+10% noise). "host DSA 가 suffix 에 해롭다" 는 잘못된 해석.

## 2. 검증 실험 (E1/E2) — `../SUB_212_optimal_dsa_6point/verify_fap.sh`

현 환경 (host WQ **enabled 그대로**) 에서 PIECEWISE 로만 부팅:

| 셀 | 사전 예측 (H-FaP) | 판정 |
|---|---:|---|
| E1 vanilla+PIECEWISE mix | **~8,850** (TSK_042 재현) | 적중 시 host DSA 효과 = 0 확정 |
| E2 suffix+PIECEWISE mix | **~27,851** | suffix OFF→ON "−12%" 도 FaP 부작용으로 재귀속 |

## 3. Lever — FaP×suffix 양립: Uniform Draft Padding

### 원리

- rejection sampler 는 draft 불일치 토큰을 **항상 기각** → 패딩 토큰은 정확도 무손실
  (greedy/sampling 모두 출력 동등).
- 모든 propose() 결과를 정확히 `num_speculative_tokens` 길이로 pad/truncate 하면
  순수 decode batch 가 uniform 조건을 만족 → **FULL cudagraph 적중**.
- 비용: (K − match_len) 위치의 verify FLOPs 낭비. B200 decode 는 launch-bound
  (SUB_201 §5: Qwen-7B launch 36%, Llama-70B memcpy+launch) 이므로
  FULL graph 의 launch 제거 이득 (+33~36% on vanilla) 이 우세할 것으로 예측.

### 구현

`vllm/v1/spec_decode/suffix_decoding.py` — env `VLLM_SUFFIX_PAD_UNIFORM=1`
(flag file `/tmp/vllm_l3_VLLM_SUFFIX_PAD_UNIFORM.flag` fallback, spawn-안전).
기본 OFF = upstream 동일.

### 크기 제약 (conc=32)

| K | uniform num_tokens = 32×(1+K) | capture ≤ 512? |
|---:|---:|---|
| 8 | 288 | ✓ (capture size 288 존재) |
| 15 | 512 | ✓ (정확히 상한) |
| 32 | 1,056 | ✗ (`max_cudagraph_capture_size` 인상 필요 + verify 비용 33×) |

### 셀 설계 (Llama-8B, mix, conc=32, n=1)

| 셀 | K | pad | cudagraph | 분리 목적 |
|---|---:|:---:|---|---|
| P1 | 8 | ✓ | FaP | **본명제** — FULL 적중 시 suffix 위 FaP 가산 |
| P2 | 15 | ✓ | FaP | K 상한 (512 정확) 에서의 trade-off |
| P3 | 8 | ✗ | FaP | K 축소 단독 효과 (FULL 미적중 대조군) |
| P4 | 8 | ✗ | PIECEWISE | K 축소 + PIECEWISE 기준점 |

기준점: suf K=32 PIECEWISE = **27,851** (TSK_042) / suf K=32 FaP = 24,407 (SUB_212).

### 사전 예측 (IDE_023 P0 commit)

- P1 ≈ suffix step 의 launch-bound 부분에 FaP 이득 적용 − K=8 cap 의 α tail 손실
  → **+10~25% vs 27,851** (30.6k~34.8k) 시 GO.
- P3 ≈ P4 (FULL 미적중이므로 FaP 무관) — P3 > P4 면 모델 오류 신호.
- kill: P1 < 27,851 (uniform padding 의 verify 낭비가 launch 이득 초과).

## 4. 산출물

- `sub213_sweep.sh` — E1/E2 + P1~P4 chain runner
- `runs/` — summ JSON + boot/bench logs
- `MEASUREMENTS.md` — 결과 (TBD)
