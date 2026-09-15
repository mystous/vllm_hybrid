# IDE_068 — CPU MoE 오프로딩의 서버 한계: 최대 모델 크기와 GPU 최소 장수

> 부모 `IDE_023`. 계획 `PLN_010`. 작업 `TSK_050`(실험 B) / `TSK_051`(실험 A). 게이트 `TST_024` / `TST_025`.
> 노드 violet-h100-016 — H100 80 GB ×8 / Xeon Platinum 8480+ ×2 (AMX) / DDR5 2 TB / **turbo OFF 2.0 GHz 고정**.
> 브랜치 `feat/moe-offload-limits`. 진행 로그 `PROGRESS.md`, 결과 `RESULT.md`.

## 질문

| | 질문 | 대상 |
|---|---|---|
| **A** | 이 서버가 CPU MoE 오프로딩으로 지원할 수 있는 **최대 모델**은 무엇인가 | 공개 최대 MoE `moonshotai/Kimi-K2-Instruct` (1.03 TB FP8 · DeepseekV3 arch · 61층 · 384+1 experts · top-8) |
| **B** | GPU-only 로 **8장이 필요한 가장 큰 보유 모델**을 오프로딩하면 GPU 를 **몇 장까지** 줄일 수 있는가 | `Qwen3-Coder-480B-A35B-Instruct-FP8` (450 GB · 62층 · 160 experts · top-8). TP=4 GPU-only OOM 은 `TSK_047` 실증 |

## 배경 — 왜 이 두 질문인가

`IDE_023`/`TSK_043` 은 GPU-only 로 적재 불가한 모델(R1-0528, 642 GB)이 expert 를 CPU 로 내리면 서빙된다는 것을 보였다.
그 결과에서 자연히 따라오는 두 한계 질문이 이것이다 — 위로는 DRAM 2 TB 가 허용하는 최대 모델, 아래로는 attention 만 남은 GPU 가 몇 장까지 줄어드는가.

480B 의 비-expert 파라미터(attention·router·embedding)는 약 12 B 로 FP8 기준 12 GB 수준이다. 산술로는 GPU 1장에도 들어간다. 실측으로 확인해야 하는 것은 (1) 실제로 부팅·서빙되는가 (2) TP 를 줄일 때 처리량이 어떻게 변하는가 (3) 품질이 유지되는가다.

## 원칙 (선행 캠페인 승계)

- "CPU busy" 는 성공 지표가 아니다. binding 지표는 처리량·TTFT·TPOT·품질이다.
- 성립 = 전 요청 완료 + greedy 4문항 정상. 하나라도 빠지면 성립이 아니다.
- 모든 CPU 수치는 turbo OFF 하한이다. 매 표에 명시한다.
- 수치 없는 이득 주장 금지. 원본 로그는 `eval/results/<TS>_ide068_*/` 에 남긴다.
- 저장소 vLLM fork 소스는 바꾸지 않는다. 컨테이너-로컬 패치와 플래그만 쓴다.

## 스택

SGLang 0.5.18 (`lmsysorg/sglang:latest`) + kt-kernel 0.7.0.post1 (컨테이너 `sgl-kt`, 패치 적용 상태) · 벤치 `vllm bench serve --backend openai` (컨테이너 `vllm-h100`).
