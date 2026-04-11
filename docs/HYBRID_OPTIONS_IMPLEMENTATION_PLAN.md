# CPU+GPU 하이브리드 최적화 구현 계획서

> **목표**: GPU-only 대비 더 높은 처리량을 달성하는 CPU+GPU 병렬 배치 추론
> **마지막 업데이트**: 2026-04-11
> **설계의 단일 진실 공급원**: `docs/paper/main.tex`

---

## 개요

### 두 가지 옵션

| 옵션 | 구성 | 대상 환경 |
|------|------|----------|
| **Option A (parallel-batch)** | Dual-process CapacityAwareRouter | Dense 모델 전반 (구현 완료) |
| **Option B (moe-hybrid)** | MoE Expert Offload + N-gram + Disaggregated | MoE 모델 (미래) |

현재 구현된 것은 **parallel-batch** 옵션 한 가지이며, GPU engine 과 CPU engine 을
별도 OS 프로세스로 띄워 `HybridAsyncMPClient` 가 ZMQ identity 기반으로 라우팅한다.
MoE 계열은 `expert_offload.py` / `ngram_proposer_dynamic.py` / `disaggregated/`
디렉토리에 스켈레톤만 존재한다.

### 실행 예시

```bash
# parallel-batch (구현 완료). 모든 CPU 파라미터는 auto 권장
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --hybrid-mode parallel-batch

# 명시적 override (디버깅용)
vllm serve meta-llama/Llama-3-70B \
  --tensor-parallel-size 8 \
  --hybrid-mode parallel-batch \
  --hybrid-routing-strategy capacity \
  --hybrid-routing-priority cpu-first

# moe-hybrid (미래 계획)
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --hybrid-mode moe-hybrid
```

### 구현 상태 (2026-04-11)

| 옵션 | 상태 | 설명 |
|------|------|------|
| **parallel-batch** | 구현 완료 | Dual-process, CapacityAwareRouter, IPEX/AVX-512/AMX graceful fallback |
| **moe-hybrid** | 스켈레톤 | MoE Expert Offload / Speculative / Disaggregated (미래) |

---

## 목차

1. [아키텍처 개요](#1-아키텍처-개요)
2. [Option A 상세 설계](#2-option-a-상세-설계)
3. [Option B 상세 설계](#3-option-b-상세-설계)
4. [공통 인프라](#4-공통-인프라)
5. [구현 로드맵](#5-구현-로드맵)
6. [파일 구조](#6-파일-구조)
7. [API 설계](#7-api-설계)
8. [테스트 계획](#8-테스트-계획)

---

## 1. 아키텍처 개요

### 1.1 Option A 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Option A 아키텍처                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    [3] Disaggregated Serving                     │   │
│  │  ┌───────────────────┐         ┌───────────────────────────┐    │   │
│  │  │   Prefill Node    │   KV    │      Decode Node          │    │   │
│  │  │   (GPU/CPU Pool)  │ ─────→  │      (GPU + CPU)          │    │   │
│  │  │                   │ Cache   │                           │    │   │
│  │  │  - 긴 프롬프트    │         │  ┌─────────────────────┐  │    │   │
│  │  │  - 배치 처리      │         │  │ [1] MoE Offload     │  │    │   │
│  │  │  - 높은 처리량    │         │  │ GPU: Attention      │  │    │   │
│  │  └───────────────────┘         │  │ CPU: Experts        │  │    │   │
│  │                                │  └─────────────────────┘  │    │   │
│  │                                │            +              │    │   │
│  │                                │  ┌─────────────────────┐  │    │   │
│  │                                │  │ [2] N-gram Lookup   │  │    │   │
│  │                                │  │ CPU: 패턴 매칭      │  │    │   │
│  │                                │  │ GPU: 검증           │  │    │   │
│  │                                │  └─────────────────────┘  │    │   │
│  │                                └───────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Option B 아키텍처 (parallel-batch, 구현 완료)

실제 구현은 **Dual-Process CapacityAwareRouter** 이다. 아래 "APEX Scheduler" 그림은
초기 계획이고, 현재는 GPU/CPU engine 을 별도 OS 프로세스로 띄우고 `HybridAsyncMPClient`
가 ZMQ identity 기반으로 routing 한다. 단일 프로세스 내 partition/merger 는 존재하지
않는다. 상세는 `CLAUDE.md` 및 `docs/paper/main.tex` §3 참조.

```
HybridAsyncMPClient (단일 API endpoint)
│
├─ CapacityAwareRouter (cpu_in_flight < cpu_max_num_seqs ? CPU : GPU)
├─ input socket (ZMQ ROUTER, identity dispatch)
│   ├─ GPU engine: identity = b'\x00\x00'
│   └─ CPU engine: identity = b'\x01\x00'  (multi-NUMA 시 \x02\x00 …)
└─ output socket (ZMQ PULL)

GPU EngineCoreProc [별도 PID] → MultiprocExecutor → N × GPUWorker
CPU EngineCoreProc [별도 PID, num_numa 개] → UniProcExecutor → CPUWorker
  ├─ init_cpu_threads_env (C++, _C_utils): OMP 1:1 pin + numa strict membind
  └─ IPEX → oneDNN → AMX/AVX-512/AVX2 graceful fallback
```

핵심 원칙:
- `num_cpu_engines = num_numa_nodes` auto
- per-engine `cpu_max_num_seqs = 1` 고정 — 1 시퀀스가 NUMA 의 모든 물리 코어를 OMP 로 점유
- `cpu-first` routing priority (논문 Property 2: GPU non-interference)
- 3 차원 독립 fallback chain (attention kernel / thread bind / ISA)

--- (아래는 초기 계획 그림, 참고용으로 남겨둠) ---

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Option B 아키텍처                              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      APEX Scheduler                              │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │              Profiler + Dynamic Partitioner                │  │   │
│  │  │  - 배치별 최적 CPU/GPU 비율 결정                           │  │   │
│  │  │  - 런타임 조정                                             │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  │                              ↓                                   │   │
│  │  ┌─────────────────────┐    ┌─────────────────────────────┐     │   │
│  │  │    GPU Worker       │    │       CPU Worker            │     │   │
│  │  │  ┌───────────────┐  │    │  ┌───────────────────────┐  │     │   │
│  │  │  │ Batch A, C, E │  │    │  │ Batch B, D, F         │  │     │   │
│  │  │  │ (전체 모델)   │  │    │  │ (전체 모델, INT8)     │  │     │   │
│  │  │  └───────────────┘  │    │  └───────────────────────┘  │     │   │
│  │  │  처리량: 80%        │    │  처리량: 20%                │     │   │
│  │  └─────────────────────┘    └─────────────────────────────┘     │   │
│  │                              ↓                                   │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │                    Result Merger                           │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Option A 상세 설계

### 2.1 컴포넌트 1: MoE Expert Offload

#### 2.1.1 개념

```
MoE 레이어 구조:
┌─────────────────────────────────────────────────────────────┐
│  Input → Router → [Expert 0] [Expert 1] ... [Expert N] → Output
│                        ↑          ↑              ↑
│                    활성화      비활성화       비활성화
│                   (GPU 실행)  (CPU 대기)     (CPU 대기)
└─────────────────────────────────────────────────────────────┘

- Router가 Top-K expert 선택 (보통 K=2 또는 K=8)
- 선택된 expert만 계산 필요
- 나머지는 메모리만 차지
```

#### 2.1.2 구현 전략

```python
# vllm/model_executor/layers/moe/expert_offload.py

class ExpertOffloadManager:
    """MoE Expert CPU-GPU 오프로드 관리자"""

    def __init__(
        self,
        num_experts: int,
        expert_size: int,
        num_gpu_experts: int,  # GPU에 상주할 expert 수
        cpu_dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_experts = num_experts
        self.num_gpu_experts = num_gpu_experts

        # GPU에 상주할 "hot" experts (자주 사용되는 것)
        self.gpu_experts: Dict[int, nn.Module] = {}

        # CPU에 대기할 experts
        self.cpu_experts: Dict[int, nn.Module] = {}

        # Expert 사용 통계 (LRU 캐시용)
        self.expert_usage_count: Dict[int, int] = defaultdict(int)

        # CPU-GPU 전송 스트림
        self.transfer_stream = torch.cuda.Stream()

    def route_and_compute(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """라우팅 후 expert 계산"""

        # 1. 라우터로 expert 선택
        routing_weights, selected_experts = self._compute_routing(router_logits)

        # 2. 선택된 expert 분류
        gpu_experts, cpu_experts = self._classify_experts(selected_experts)

        # 3. GPU expert 계산 (즉시)
        gpu_output = self._compute_gpu_experts(
            hidden_states, routing_weights, gpu_experts
        )

        # 4. CPU expert 계산 (병렬)
        cpu_output = self._compute_cpu_experts(
            hidden_states, routing_weights, cpu_experts
        )

        # 5. 결과 합산
        return gpu_output + cpu_output.to(hidden_states.device)

    def _compute_cpu_experts(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: List[int],
    ) -> torch.Tensor:
        """CPU에서 expert 계산 (AVX-512 최적화)"""

        # GPU → CPU 전송
        hidden_cpu = hidden_states.to('cpu', non_blocking=True)

        # CPU에서 계산 (병렬)
        outputs = []
        with ThreadPoolExecutor(max_workers=len(expert_indices)) as executor:
            futures = []
            for idx in expert_indices:
                expert = self.cpu_experts[idx]
                weight = routing_weights[:, idx]
                futures.append(
                    executor.submit(self._run_expert_cpu, expert, hidden_cpu, weight)
                )

            for future in futures:
                outputs.append(future.result())

        # 합산
        result = torch.stack(outputs).sum(dim=0)
        return result

    def _run_expert_cpu(
        self,
        expert: nn.Module,
        hidden: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """단일 expert CPU 실행 (AVX-512 VNNI 최적화)"""

        with torch.no_grad():
            # INT8 양자화된 expert 사용 시
            if hasattr(expert, 'int8_forward'):
                output = expert.int8_forward(hidden)
            else:
                output = expert(hidden)

            return output * weight.unsqueeze(-1)

    def update_expert_cache(self, selected_experts: torch.Tensor):
        """Expert 캐시 업데이트 (LRU 기반)"""

        # 사용 통계 업데이트
        unique_experts = selected_experts.unique().tolist()
        for idx in unique_experts:
            self.expert_usage_count[idx] += 1

        # 주기적으로 GPU expert 교체
        if self._should_swap():
            self._swap_experts()

    def _swap_experts(self):
        """사용 빈도 기반 expert 교체"""

        # 가장 자주 사용되는 expert를 GPU로
        sorted_experts = sorted(
            self.expert_usage_count.items(),
            key=lambda x: x[1],
            reverse=True
        )

        new_gpu_experts = [idx for idx, _ in sorted_experts[:self.num_gpu_experts]]

        # 비동기 전송
        with torch.cuda.stream(self.transfer_stream):
            for idx in new_gpu_experts:
                if idx not in self.gpu_experts:
                    # CPU → GPU
                    self.gpu_experts[idx] = self.cpu_experts.pop(idx).cuda()

            for idx in list(self.gpu_experts.keys()):
                if idx not in new_gpu_experts:
                    # GPU → CPU
                    self.cpu_experts[idx] = self.gpu_experts.pop(idx).cpu()
```

#### 2.1.3 MoE 레이어 통합

```python
# vllm/model_executor/layers/moe/fused_moe_offload.py

class FusedMoEWithOffload(nn.Module):
    """CPU 오프로드를 지원하는 Fused MoE 레이어"""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        offload_config: Optional[ExpertOffloadConfig] = None,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # 라우터
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

        # 오프로드 관리자
        if offload_config and offload_config.enabled:
            self.offload_manager = ExpertOffloadManager(
                num_experts=num_experts,
                expert_size=intermediate_size,
                num_gpu_experts=offload_config.num_gpu_experts,
            )
            self._setup_offload()
        else:
            self.offload_manager = None

    def _setup_offload(self):
        """Expert 오프로드 초기 설정"""

        # 처음에는 균등 분배
        num_gpu = self.offload_manager.num_gpu_experts

        for i, expert in enumerate(self.experts):
            if i < num_gpu:
                self.offload_manager.gpu_experts[i] = expert.cuda()
            else:
                # INT8 양자화 후 CPU로
                expert_int8 = quantize_expert_to_int8(expert)
                self.offload_manager.cpu_experts[i] = expert_int8.cpu()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward with optional offload"""

        # 라우팅
        router_logits = self.gate(hidden_states)

        if self.offload_manager:
            # 오프로드 모드
            return self.offload_manager.route_and_compute(
                hidden_states, router_logits
            )
        else:
            # 일반 모드
            return self._standard_moe_forward(hidden_states, router_logits)
```

### 2.2 컴포넌트 2: N-gram Lookahead Decoding

#### 2.2.1 개념

```
N-gram Lookahead:
┌─────────────────────────────────────────────────────────────────┐
│  1. CPU: 이전 출력에서 N-gram 패턴 매칭                          │
│     "The quick brown" → 과거에 "The quick brown fox" 출력한 적 있음
│     → ["fox", "jumps", "over"] 추측                              │
│                                                                  │
│  2. GPU: 추측 토큰 검증 (한 번의 forward)                        │
│     Input: "The quick brown" + ["fox", "jumps", "over"]          │
│     Output: [✓ fox] [✓ jumps] [✗ over → "the"]                  │
│     → 2개 토큰 즉시 채택!                                        │
│                                                                  │
│  효과: 3개 토큰을 1번의 forward로 처리 (vs 3번)                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.2 구현

```python
# vllm/spec_decode/ngram_proposer.py

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional
import threading

class NGramProposer:
    """N-gram 기반 추측 토큰 제안자 (CPU에서 실행)"""

    def __init__(
        self,
        n: int = 3,                    # N-gram 크기
        num_speculative_tokens: int = 5,  # 추측할 토큰 수
        min_frequency: int = 2,        # 최소 출현 빈도
    ):
        self.n = n
        self.num_speculative_tokens = num_speculative_tokens
        self.min_frequency = min_frequency

        # N-gram 저장소: (n-1)-gram → {next_token: count}
        self.ngram_store: Dict[Tuple[int, ...], Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # 락 (멀티스레드 안전)
        self.lock = threading.RLock()

        # 통계
        self.total_proposals = 0
        self.accepted_proposals = 0

    def update(self, token_ids: List[int]):
        """출력 토큰으로 N-gram 업데이트 (백그라운드)"""

        with self.lock:
            for i in range(len(token_ids) - self.n + 1):
                prefix = tuple(token_ids[i:i + self.n - 1])
                next_token = token_ids[i + self.n - 1]
                self.ngram_store[prefix][next_token] += 1

    def propose(
        self,
        context_tokens: List[int],
    ) -> Tuple[List[int], List[float]]:
        """N-gram 기반 추측 토큰 제안"""

        proposals = []
        confidences = []

        current_context = list(context_tokens)

        with self.lock:
            for _ in range(self.num_speculative_tokens):
                # (n-1)-gram prefix 추출
                if len(current_context) >= self.n - 1:
                    prefix = tuple(current_context[-(self.n - 1):])
                else:
                    prefix = tuple(current_context)

                # 다음 토큰 예측
                if prefix in self.ngram_store:
                    candidates = self.ngram_store[prefix]

                    # 빈도 기반 정렬
                    sorted_candidates = sorted(
                        candidates.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    if sorted_candidates and sorted_candidates[0][1] >= self.min_frequency:
                        next_token = sorted_candidates[0][0]
                        total_count = sum(c for _, c in sorted_candidates)
                        confidence = sorted_candidates[0][1] / total_count

                        proposals.append(next_token)
                        confidences.append(confidence)
                        current_context.append(next_token)
                        continue

                # 매칭 실패 시 중단
                break

        self.total_proposals += len(proposals)
        return proposals, confidences

    def record_acceptance(self, num_accepted: int):
        """채택된 토큰 수 기록 (통계용)"""
        self.accepted_proposals += num_accepted

    @property
    def acceptance_rate(self) -> float:
        """채택률"""
        if self.total_proposals == 0:
            return 0.0
        return self.accepted_proposals / self.total_proposals


class NGramLookaheadWorker:
    """N-gram Lookahead 워커 (CPU 스레드에서 실행)"""

    def __init__(
        self,
        proposer: NGramProposer,
        tokenizer,
    ):
        self.proposer = proposer
        self.tokenizer = tokenizer

        # 비동기 업데이트 큐
        self.update_queue = queue.Queue()

        # 백그라운드 업데이트 스레드
        self.update_thread = threading.Thread(
            target=self._background_update,
            daemon=True
        )
        self.update_thread.start()

    def _background_update(self):
        """백그라운드에서 N-gram 업데이트"""
        while True:
            try:
                token_ids = self.update_queue.get(timeout=1.0)
                self.proposer.update(token_ids)
            except queue.Empty:
                continue

    def get_proposals(
        self,
        context_token_ids: List[int],
    ) -> List[int]:
        """추측 토큰 반환 (CPU에서 마이크로초 단위로 실행)"""

        proposals, _ = self.proposer.propose(context_token_ids)
        return proposals

    def submit_output(self, output_token_ids: List[int]):
        """출력 토큰을 업데이트 큐에 추가"""
        self.update_queue.put(output_token_ids)
```

#### 2.2.3 vLLM Speculative Decoding 통합

```python
# vllm/spec_decode/ngram_spec_worker.py

from vllm.spec_decode.interfaces import SpeculativeProposer
from vllm.spec_decode.ngram_proposer import NGramProposer, NGramLookaheadWorker

class NGramSpeculativeWorker(SpeculativeProposer):
    """N-gram 기반 Speculative Decoding 워커"""

    def __init__(
        self,
        proposer_config: NGramProposerConfig,
        tokenizer,
    ):
        self.proposer = NGramProposer(
            n=proposer_config.n,
            num_speculative_tokens=proposer_config.num_speculative_tokens,
        )
        self.worker = NGramLookaheadWorker(self.proposer, tokenizer)

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> SpeculativeProposals:
        """추측 토큰 제안"""

        proposals = []

        for seq_group in execute_model_req.seq_group_metadata_list:
            seq_data = seq_group.seq_data
            context_tokens = list(seq_data.get_token_ids())

            # CPU에서 N-gram 매칭 (매우 빠름)
            proposed_tokens = self.worker.get_proposals(context_tokens)

            proposals.append(proposed_tokens)

        return SpeculativeProposals(
            proposal_token_ids=proposals,
            proposal_probs=None,  # N-gram은 확률 없음
            proposal_lens=[len(p) for p in proposals],
        )

    def update_from_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sampler_output: SamplerOutput,
    ):
        """출력으로 N-gram 테이블 업데이트"""

        for seq_group, output in zip(
            execute_model_req.seq_group_metadata_list,
            sampler_output.outputs
        ):
            output_tokens = [o.token_id for o in output.samples]
            self.worker.submit_output(output_tokens)
```

### 2.3 컴포넌트 3: Disaggregated Serving

#### 2.3.1 개념

```
Disaggregated Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Prefill Pool                    Decode Pool                    │
│  ┌──────────────┐               ┌──────────────────────┐        │
│  │ GPU 0        │               │ GPU 4                │        │
│  │ GPU 1        │    KV Cache   │ GPU 5                │        │
│  │ (또는 CPU)   │ ───────────→  │ GPU 6                │        │
│  │              │    Transfer   │ GPU 7                │        │
│  └──────────────┘               └──────────────────────┘        │
│        ↑                              ↓                          │
│  긴 프롬프트 처리                짧은 응답 생성                   │
│  (처리량 최적화)                (지연시간 최적화)                 │
│                                                                  │
│  특성:                          특성:                            │
│  - Compute bound               - Memory bound                   │
│  - 배치 효율적                 - 배치 비효율적                   │
│  - Latency tolerance 높음      - Latency critical               │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 구현

```python
# vllm/engine/disaggregated/prefill_node.py

class PrefillNode:
    """Prefill 전용 노드"""

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,  # GPU 또는 CPU
        kv_transfer_config: KVTransferConfig,
    ):
        self.model_config = model_config
        self.device = device_config.device

        # 모델 로드 (Prefill 최적화)
        self.model = self._load_model_for_prefill()

        # KV Cache 전송 클라이언트
        self.kv_sender = KVCacheSender(kv_transfer_config)

    def _load_model_for_prefill(self):
        """Prefill 최적화된 모델 로드"""

        model = load_model(self.model_config)

        if self.device == 'cpu':
            # CPU Prefill: INT8 양자화 + AVX-512
            model = quantize_for_cpu_prefill(model)
            model = model.to('cpu')
        else:
            # GPU Prefill: Flash Attention 활성화
            model = optimize_for_gpu_prefill(model)
            model = model.cuda()

        return model

    async def run_prefill(
        self,
        request: PrefillRequest,
    ) -> PrefillResult:
        """Prefill 실행 및 KV Cache 전송"""

        # 1. Prefill 실행
        with torch.no_grad():
            hidden_states, kv_cache = self.model.prefill(
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
            )

        # 2. KV Cache를 Decode 노드로 전송
        kv_transfer_handle = await self.kv_sender.send_async(
            kv_cache=kv_cache,
            dest_node=request.decode_node_id,
            request_id=request.request_id,
        )

        return PrefillResult(
            request_id=request.request_id,
            kv_transfer_handle=kv_transfer_handle,
            num_tokens=request.input_ids.shape[1],
        )


# vllm/engine/disaggregated/decode_node.py

class DecodeNode:
    """Decode 전용 노드 (Option A의 MoE Offload + N-gram 포함)"""

    def __init__(
        self,
        model_config: ModelConfig,
        moe_offload_config: Optional[ExpertOffloadConfig],
        ngram_config: Optional[NGramProposerConfig],
        kv_transfer_config: KVTransferConfig,
    ):
        self.model_config = model_config

        # 모델 로드 (MoE Offload 적용)
        self.model = self._load_model_with_offload(moe_offload_config)

        # N-gram Proposer
        if ngram_config:
            self.ngram_worker = NGramSpeculativeWorker(ngram_config, self.tokenizer)
        else:
            self.ngram_worker = None

        # KV Cache 수신
        self.kv_receiver = KVCacheReceiver(kv_transfer_config)

    def _load_model_with_offload(self, offload_config):
        """MoE Offload 적용된 모델 로드"""

        model = load_model(self.model_config)

        if offload_config and offload_config.enabled:
            # MoE 레이어에 오프로드 적용
            for name, module in model.named_modules():
                if isinstance(module, MoELayer):
                    offloaded = FusedMoEWithOffload(
                        num_experts=module.num_experts,
                        top_k=module.top_k,
                        hidden_size=module.hidden_size,
                        intermediate_size=module.intermediate_size,
                        offload_config=offload_config,
                    )
                    # 교체
                    set_module_by_name(model, name, offloaded)

        return model.cuda()

    async def run_decode(
        self,
        request: DecodeRequest,
    ) -> DecodeResult:
        """Decode 실행 (N-gram Speculative 포함)"""

        # 1. KV Cache 수신 대기
        kv_cache = await self.kv_receiver.receive_async(request.request_id)

        # 2. Decode 루프
        output_tokens = []

        while not self._should_stop(request, output_tokens):
            # N-gram 추측 (CPU)
            if self.ngram_worker:
                spec_tokens = self.ngram_worker.get_spec_proposals(
                    context_tokens=request.input_ids + output_tokens
                )
            else:
                spec_tokens = []

            # Forward (GPU, 추측 토큰 포함)
            if spec_tokens:
                # Speculative forward
                logits = self.model.forward_speculative(
                    input_ids=output_tokens[-1:] + spec_tokens,
                    kv_cache=kv_cache,
                )

                # 검증
                accepted, rejected_idx = self._verify_speculative(
                    logits, spec_tokens
                )
                output_tokens.extend(accepted)

                # N-gram 업데이트
                self.ngram_worker.update_from_output(accepted)
            else:
                # 일반 forward
                logits = self.model.forward(
                    input_ids=output_tokens[-1:],
                    kv_cache=kv_cache,
                )
                next_token = self._sample(logits)
                output_tokens.append(next_token)

        return DecodeResult(
            request_id=request.request_id,
            output_tokens=output_tokens,
        )


# vllm/engine/disaggregated/coordinator.py

class DisaggregatedCoordinator:
    """Prefill/Decode 노드 조율자"""

    def __init__(
        self,
        prefill_nodes: List[PrefillNode],
        decode_nodes: List[DecodeNode],
    ):
        self.prefill_nodes = prefill_nodes
        self.decode_nodes = decode_nodes

        # 로드 밸런서
        self.prefill_lb = LoadBalancer(prefill_nodes)
        self.decode_lb = LoadBalancer(decode_nodes)

    async def process_request(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """요청 처리"""

        # 1. Prefill 노드 선택 및 실행
        prefill_node = self.prefill_lb.select()
        prefill_result = await prefill_node.run_prefill(
            PrefillRequest(
                request_id=request.id,
                input_ids=request.input_ids,
                decode_node_id=self._select_decode_node().id,
            )
        )

        # 2. Decode 노드에서 생성
        decode_node = self.decode_lb.select()
        decode_result = await decode_node.run_decode(
            DecodeRequest(
                request_id=request.id,
                input_ids=request.input_ids,
                max_tokens=request.max_tokens,
            )
        )

        return InferenceResponse(
            request_id=request.id,
            output_tokens=decode_result.output_tokens,
        )
```

---

## 3. Option B (parallel-batch) 상세 설계

> 실제 구현은 "APEX 스케줄러 + cpu_ratio" 가 아니라 **CapacityAwareRouter + Dual Process**
> 이다. 아래 3.1 의 profiling/ratio 계산 섹션들은 **구식 계획** 이며, 현재 코드는
> `vllm/v1/engine/hybrid_core.py :: CapacityAwareRouter` 의 3 전략 (capacity /
> length-aware / throughput-adaptive) 만 사용한다. 자세한 설계는 논문 §3.3 Algorithm 1/2/3 참조.

### 3.1 CapacityAwareRouter (현재 구현)

```
매 요청 도착 시:
    if cpu_in_flight < cpu_max_num_seqs:   # 기본값: N = num_numa_nodes × 1
        route → CPU (identity b'\x01\x00')
    else:
        route → GPU (identity b'\x00\x00')
```

- `cpu_max_num_seqs = 1` per engine 이므로 CPU 는 한 번에 1 seq × NUMA count 만 처리
- nominal capacity 초과 시 즉시 GPU 로 overflow → CPU 가 GPU 를 블로킹하지 않음 (Property 2)
- length-aware / throughput-adaptive 는 prompt length 또는 측정된 EMA 처리량으로 threshold 를 조정
- `round-robin` 은 디버깅용

### 3.1 (legacy) APEX 스케줄러 — 초기 계획 (구현되지 않음)

---

## 3.2 자동 프로파일링 상세 설계

### 3.2.1 현재 구현 상태 (2026-02-03)

| 항목 | 구현 방식 | 상태 |
|------|----------|------|
| GPU 처리량 | 휴리스틱 (고정값 100 tok/s) | ⚠️ 추정값 |
| CPU 처리량 | 실제 측정 (더미 입력 추론) | ✅ 구현됨 |
| 비율 계산 | 처리량 비례 분배 | ✅ 구현됨 |

### 3.2.2 비율 계산 공식

```
최적 비율 계산:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. 처리량 측정                                                  │
│     T_gpu = GPU 처리량 (tok/s)                                  │
│     T_cpu = CPU 처리량 (tok/s)                                  │
│                                                                  │
│  2. 비율 계산 (처리량 비례 분배)                                 │
│     R_gpu = T_gpu / (T_gpu + T_cpu)                             │
│     R_cpu = T_cpu / (T_gpu + T_cpu)                             │
│                                                                  │
│  3. 예시 (임의의 x86_64 + NVIDIA GPU)                           │
│     T_gpu = 100 tok/s (추정)                                    │
│     T_cpu = 5 tok/s (측정)                                      │
│     R_gpu = 100 / 105 = 95.2%                                   │
│     R_cpu = 5 / 105 = 4.8%                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2.3 CPU 처리량 측정 방법 (구현됨)

```python
def _measure_cpu_throughput(self, num_batches: int = 5) -> float:
    """CPU 처리량 실제 측정."""

    # 1. 더미 입력 생성
    dummy_input = torch.randint(0, 32000, (batch_size, seq_len), device='cpu')

    # 2. 워밍업 (2회)
    for _ in range(2):
        with torch.no_grad():
            _ = self.cpu_worker.model(dummy_input)

    # 3. 실제 측정
    start = time.perf_counter()
    for _ in range(num_batches):
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=use_bf16):
            _ = self.cpu_worker.model(dummy_input)
    elapsed = time.perf_counter() - start

    # 4. 처리량 계산
    total_tokens = num_batches * seq_len * batch_size
    throughput = total_tokens / elapsed  # tok/s

    return throughput
```

### 3.2.4 GPU 처리량 측정 (TODO - 개선 필요)

현재 GPU 처리량은 **휴리스틱 추정값**을 사용합니다:

```python
def _measure_gpu_throughput(self, num_batches: int) -> float:
    """GPU 처리량 측정 - 현재는 추정값 사용."""

    # TODO: 실제 GPU executor로 측정 구현 필요
    # 현재는 H100 기준 추정값 사용
    estimated_gpu_throughput = 100.0  # tok/s per sequence

    return estimated_gpu_throughput
```

**개선 계획:**

```python
def _measure_gpu_throughput(self, num_batches: int) -> float:
    """GPU 처리량 실제 측정 (개선 버전)."""

    if self.gpu_executor is None:
        return 0.0

    # 1. 더미 요청 생성
    dummy_requests = create_dummy_requests(
        num_seqs=self.profile_batch_size,
        seq_len=self.profile_seq_len,
    )

    # 2. 워밍업
    for _ in range(2):
        self.gpu_executor.execute_model(dummy_requests)

    # 3. 측정 (CUDA 동기화 포함)
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_batches):
        self.gpu_executor.execute_model(dummy_requests)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # 4. 처리량 계산
    total_tokens = num_batches * self.profile_seq_len * self.profile_batch_size
    throughput = total_tokens / elapsed

    return throughput
```

### 3.2.5 고급 프로파일링 (향후 구현)

```
고급 프로파일링 전략:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. 다중 배치 크기 프로파일링                                    │
│     batch_sizes = [1, 4, 8, 16, 32]                             │
│     각 배치 크기에서 GPU/CPU 처리량 측정                         │
│     → 배치 크기별 최적 비율 테이블 생성                          │
│                                                                  │
│  2. 시퀀스 길이별 프로파일링                                     │
│     seq_lens = [128, 512, 1024, 2048]                           │
│     긴 시퀀스 → GPU 선호 (compute bound)                        │
│     짧은 시퀀스 → CPU 가능 (memory bound)                       │
│                                                                  │
│  3. 동적 비율 조정                                               │
│     런타임에 실제 처리량 모니터링                                │
│     편차 발생 시 비율 재조정                                     │
│                                                                  │
│  4. 메모리 오버헤드 고려                                         │
│     CPU 모델 메모리 사용량                                       │
│     GPU VRAM 여유 공간                                          │
│     KV Cache 크기                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2.6 프로파일링 설정 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `profile_seq_len` | 128 | 프로파일링용 시퀀스 길이 |
| `profile_batch_size` | 1 | 프로파일링용 배치 크기 |
| `profile_num_batches` | 5 | 측정 반복 횟수 |
| `warmup_iterations` | 2 | 워밍업 반복 횟수 |

### 3.2.7 수동 비율 지정 vs 자동 프로파일링

```bash
# 자동 프로파일링 (기본값)
--hybrid-mode parallel-batch
# → 서버 시작 시 CPU/GPU 처리량 측정 후 비율 자동 결정

# 수동 비율 지정 (프로파일링 건너뜀)
--hybrid-mode parallel-batch --hybrid-cpu-ratio 0.1
# → CPU가 전체 요청의 10% 처리, 프로파일링 생략
```

**권장 사항:**
- 첫 실행: 자동 프로파일링으로 최적 비율 확인
- 운영 환경: 확인된 비율을 수동 지정하여 시작 시간 단축

#### 3.1.2 구현

```python
# vllm/executor/apex_executor.py

class APEXProfiler:
    """CPU/GPU 성능 프로파일러"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.gpu_throughput: Optional[float] = None
        self.cpu_throughput: Optional[float] = None

    def profile(self) -> Tuple[float, float]:
        """CPU/GPU 처리량 측정"""

        # GPU 프로파일링
        gpu_model = load_model(self.model_config).cuda()
        self.gpu_throughput = self._measure_throughput(gpu_model, 'cuda')
        del gpu_model
        torch.cuda.empty_cache()

        # CPU 프로파일링 (INT8 양자화)
        cpu_model = load_model(self.model_config)
        cpu_model = quantize_to_int8(cpu_model).cpu()
        self.cpu_throughput = self._measure_throughput(cpu_model, 'cpu')
        del cpu_model

        return self.gpu_throughput, self.cpu_throughput

    def _measure_throughput(
        self,
        model: nn.Module,
        device: str,
        num_warmup: int = 3,
        num_measure: int = 10,
    ) -> float:
        """처리량 측정 (tok/s)"""

        # 더미 입력
        input_ids = torch.randint(0, 32000, (1, 128)).to(device)

        # 워밍업
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model(input_ids)

        # 측정
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_measure):
            with torch.no_grad():
                _ = model(input_ids)

        if device == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # 토큰/초
        total_tokens = num_measure * 128
        return total_tokens / elapsed

    def get_optimal_ratio(self) -> Tuple[float, float]:
        """최적 CPU/GPU 배치 비율"""

        if self.gpu_throughput is None:
            self.profile()

        total = self.gpu_throughput + self.cpu_throughput
        gpu_ratio = self.gpu_throughput / total
        cpu_ratio = self.cpu_throughput / total

        return gpu_ratio, cpu_ratio


class APEXScheduler:
    """APEX 배치 스케줄러"""

    def __init__(
        self,
        gpu_ratio: float,
        cpu_ratio: float,
        max_batch_size: int = 64,
    ):
        self.gpu_ratio = gpu_ratio
        self.cpu_ratio = cpu_ratio
        self.max_batch_size = max_batch_size

    def partition_batch(
        self,
        requests: List[InferenceRequest],
    ) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """배치를 GPU/CPU로 분할"""

        n = len(requests)
        gpu_count = int(n * self.gpu_ratio)

        # GPU는 더 긴 요청 (compute bound에서 효율적)
        sorted_requests = sorted(requests, key=lambda r: len(r.input_ids), reverse=True)

        gpu_batch = sorted_requests[:gpu_count]
        cpu_batch = sorted_requests[gpu_count:]

        return gpu_batch, cpu_batch


class APEXExecutor:
    """APEX 실행기"""

    def __init__(
        self,
        model_config: ModelConfig,
        cpu_config: CPUExecutorConfig,
    ):
        self.model_config = model_config

        # 프로파일링
        profiler = APEXProfiler(model_config)
        gpu_ratio, cpu_ratio = profiler.get_optimal_ratio()

        logger.info(f"APEX optimal ratio - GPU: {gpu_ratio:.1%}, CPU: {cpu_ratio:.1%}")

        # 스케줄러
        self.scheduler = APEXScheduler(gpu_ratio, cpu_ratio)

        # GPU 워커
        self.gpu_worker = GPUWorker(model_config)

        # CPU 워커 (INT8 양자화)
        self.cpu_worker = CPUWorkerINT8(model_config, cpu_config)

        # 결과 큐
        self.result_queue = asyncio.Queue()

    async def execute_batch(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResponse]:
        """배치 실행 (CPU/GPU 병렬)"""

        # 배치 분할
        gpu_batch, cpu_batch = self.scheduler.partition_batch(requests)

        # 병렬 실행
        gpu_task = asyncio.create_task(
            self.gpu_worker.execute(gpu_batch)
        ) if gpu_batch else None

        cpu_task = asyncio.create_task(
            self.cpu_worker.execute(cpu_batch)
        ) if cpu_batch else None

        # 결과 수집
        results = []

        if gpu_task:
            gpu_results = await gpu_task
            results.extend(gpu_results)

        if cpu_task:
            cpu_results = await cpu_task
            results.extend(cpu_results)

        # 원래 순서로 정렬
        results.sort(key=lambda r: r.request_id)

        return results


class CPUWorkerINT8:
    """INT8 양자화된 CPU 워커"""

    def __init__(
        self,
        model_config: ModelConfig,
        cpu_config: CPUExecutorConfig,
    ):
        # INT8 모델 로드
        self.model = load_model(model_config)
        self.model = quantize_to_int8_avx512(self.model)
        self.model = self.model.cpu()

        # 스레드 설정
        torch.set_num_threads(cpu_config.num_threads)

        # AVX-512 최적화 환경 설정
        setup_avx512_environment()

    async def execute(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResponse]:
        """CPU에서 배치 실행"""

        # CPU 작업을 별도 스레드풀에서 실행 (async 블로킹 방지)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,  # default executor
            self._execute_sync,
            requests,
        )
        return results

    def _execute_sync(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResponse]:
        """동기 실행"""

        responses = []

        for request in requests:
            input_ids = torch.tensor([request.input_ids])

            output_tokens = []
            with torch.no_grad():
                for _ in range(request.max_tokens):
                    logits = self.model(input_ids)
                    next_token = logits[:, -1, :].argmax(dim=-1).item()
                    output_tokens.append(next_token)
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[next_token]])
                    ], dim=1)

                    if next_token == self.eos_token_id:
                        break

            responses.append(InferenceResponse(
                request_id=request.id,
                output_tokens=output_tokens,
            ))

        return responses
```

---

## 4. 공통 인프라

### 4.1 설정 클래스

```python
# vllm/config.py 확장

@dataclass
class HybridConfig:
    """하이브리드 실행 설정"""

    # 모드 선택
    mode: Literal["option-a", "option-b", "none"] = "none"

    # Option A 설정
    moe_offload: Optional[ExpertOffloadConfig] = None
    ngram_spec: Optional[NGramSpecConfig] = None
    disaggregated: Optional[DisaggregatedConfig] = None

    # Option B 설정
    apex: Optional[APEXConfig] = None

    def validate(self):
        """설정 유효성 검사"""
        if self.mode == "option-a":
            assert self.apex is None, "Option A에서는 APEX 사용 불가"
        elif self.mode == "option-b":
            assert self.moe_offload is None, "Option B에서는 MoE Offload 사용 불가"
            assert self.ngram_spec is None, "Option B에서는 N-gram Spec 사용 불가"
            assert self.disaggregated is None, "Option B에서는 Disaggregated 사용 불가"


@dataclass
class ExpertOffloadConfig:
    """MoE Expert 오프로드 설정"""
    enabled: bool = True
    num_gpu_experts: int = 8  # GPU에 상주할 expert 수
    cpu_dtype: str = "int8"   # CPU expert 데이터 타입
    swap_threshold: int = 100  # expert 교체 주기


@dataclass
class NGramSpecConfig:
    """N-gram Speculative Decoding 설정"""
    enabled: bool = True
    n: int = 3                      # N-gram 크기
    num_speculative_tokens: int = 5  # 추측 토큰 수
    min_frequency: int = 2           # 최소 출현 빈도


@dataclass
class DisaggregatedConfig:
    """Disaggregated Serving 설정"""
    enabled: bool = True
    prefill_device: str = "gpu"      # "gpu" 또는 "cpu"
    num_prefill_nodes: int = 1
    num_decode_nodes: int = 1
    kv_transfer_method: str = "rdma"  # "rdma", "tcp", "shm"


@dataclass
class APEXConfig:
    """APEX 스케줄링 설정"""
    enabled: bool = True
    auto_profile: bool = True        # 자동 프로파일링
    gpu_ratio: Optional[float] = None  # 수동 설정 시
    cpu_ratio: Optional[float] = None
    cpu_dtype: str = "int8"
    cpu_num_threads: int = 48
```

### 4.2 CLI 인터페이스 (현재 구현)

실제 구현은 `vllm/engine/arg_utils.py` 의 `EngineArgs` 에 들어 있다. 옵션은 모두
`hybrid_*` prefix 를 쓰며, 기본값이 0/auto 인 옵션은 런타임에 NUMA/CPU 감지 결과로 자동 유도된다.

| 옵션 | 기본값 | 의미 |
|------|--------|------|
| `--hybrid-mode` | `none` | `none` / `parallel-batch` / `moe-hybrid` |
| `--hybrid-num-cpu-engines` | 0 (auto) | auto 는 `num_numa_nodes` |
| `--hybrid-cpu-max-seqs` | 0 (auto) | auto 는 **1 고정** (per engine) |
| `--hybrid-cpu-kvcache-gb` | 0 (auto) | auto 는 `clamp(eff_mem × 0.4, 32, 512)` |
| `--hybrid-cpu-threads` | 0 (auto) | auto 는 NUMA 노드 물리 코어 전체 |
| `--hybrid-cpu-max-batched-tokens` | 0 (auto) | `cpu_max_num_seqs × 256` |
| `--hybrid-numa-aware` / `--no-hybrid-numa-aware` | True | NUMA 최적화 on/off |
| `--hybrid-numa-node` | auto | 특정 NUMA 노드 강제 바인딩 |
| `--hybrid-routing-strategy` | `capacity` | `capacity` / `length-aware` / `throughput-adaptive` / `round-robin` |
| `--hybrid-routing-priority` | `gpu-first` | `gpu-first` / `cpu-first` |
| `--hybrid-cpu-prefill-threshold` | 512 | length-aware / throughput-adaptive 임계 |
| `--hybrid-warmup-requests` | 10 | throughput-adaptive EMA 워밍업 |
| `--hybrid-stats-log-interval` | 50 | router 통계 로그 간격 (완료 요청 수) |

권장: 모든 CPU 파라미터는 0 (auto) 로 두고 `--hybrid-mode parallel-batch` 만 지정.
수동 override 는 허용하되 `cpu_max_num_seqs ≠ 1` 은 경고 로그 (원칙 위반 알림).

아래 `option-a` / `option-b` / `--apex-cpu-ratio` / `--moe-cpu-offload` 등은 초기
계획 draft 이며 현재 CLI 에는 존재하지 않는다. 참고용으로만 남겨둔다.

```python
# (legacy / not implemented — kept for reference)
# vllm/entrypoints/openai/cli_args.py 확장 (초기 안)

def add_hybrid_args(parser: argparse.ArgumentParser):
    """하이브리드 실행 인자 추가 (초기 계획, 현재 CLI 와 다름)"""

    group = parser.add_argument_group("Hybrid Execution Options (legacy draft)")
    group.add_argument("--hybrid-mode", choices=["option-a", "option-b", "none"], default="none")
    group.add_argument("--moe-cpu-offload", action="store_true")
    group.add_argument("--ngram-spec-decode", action="store_true")
    group.add_argument("--disaggregated-prefill", action="store_true")
    group.add_argument("--apex-cpu-ratio", type=float, default=None)
    group.add_argument("--apex-cpu-threads", type=int, default=48)
    return parser
```

### 4.3 실행 예시

```bash
#!/bin/bash

# =============================================================================
# Option A: MoE Offload + N-gram + Disaggregated (DeepSeek R1 권장)
# =============================================================================

# 전체 Option A 활성화
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --hybrid-mode option-a \
  --moe-cpu-offload \
  --moe-num-gpu-experts 8 \
  --ngram-spec-decode \
  --ngram-n 3 \
  --ngram-num-spec-tokens 5 \
  --disaggregated-prefill \
  --prefill-device gpu \
  --tensor-parallel-size 8

# MoE Offload만 (간단 버전)
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --hybrid-mode option-a \
  --moe-cpu-offload \
  --tensor-parallel-size 8

# N-gram만 추가
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --hybrid-mode option-a \
  --ngram-spec-decode \
  --tensor-parallel-size 8

# =============================================================================
# Option B: APEX 스케줄링 (Dense 모델 또는 제한된 GPU)
# =============================================================================

# 자동 프로파일링
vllm serve meta-llama/Llama-3-70B \
  --hybrid-mode option-b \
  --apex-cpu-threads 48 \
  --tensor-parallel-size 8

# 수동 비율 지정
vllm serve meta-llama/Llama-3-70B \
  --hybrid-mode option-b \
  --apex-cpu-ratio 0.2 \
  --apex-cpu-threads 48 \
  --tensor-parallel-size 8

# =============================================================================
# 비교 벤치마크
# =============================================================================

# GPU only (기준)
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --tensor-parallel-size 8

# 벤치마크 실행
python benchmarks/benchmark_serving.py \
  --backend openai \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --num-prompts 500 \
  --random-input-len 128 \
  --random-output-len 128
```

---

## 5. 구현 로드맵

### 5.1 Phase 1: 기반 인프라 (1주)

```
Week 1:
├── Day 1-2: 설정 클래스 및 CLI 인터페이스
│   ├── HybridConfig, ExpertOffloadConfig 등
│   └── argparse 확장
│
├── Day 3-4: CPU INT8 양자화 파이프라인
│   ├── AVX-512 VNNI 커널 (Phase 1에서 구현한 것 활용)
│   └── 모델 양자화 함수
│
└── Day 5-7: 테스트 프레임워크
    ├── 단위 테스트
    └── 통합 테스트 스켈레톤
```

### 5.2 Phase 2: Option A - MoE Offload (1.5주)

```
Week 2-3:
├── Day 1-3: ExpertOffloadManager 구현
│   ├── GPU/CPU expert 분리
│   ├── LRU 캐시 로직
│   └── 비동기 전송
│
├── Day 4-6: FusedMoEWithOffload 구현
│   ├── 기존 MoE 레이어 래핑
│   ├── 라우팅 + 오프로드 통합
│   └── CPU expert INT8 실행
│
└── Day 7-10: 통합 및 테스트
    ├── DeepSeek R1 모델 테스트
    └── 성능 벤치마크
```

### 5.3 Phase 3: Option A - N-gram Lookahead (1주)

```
Week 4:
├── Day 1-2: NGramProposer 구현
│   ├── N-gram 저장소
│   ├── 패턴 매칭 로직
│   └── 업데이트 로직
│
├── Day 3-4: vLLM Speculative Decoding 통합
│   ├── NGramSpeculativeWorker
│   └── 검증 로직
│
└── Day 5-7: 테스트 및 튜닝
    ├── 코드 생성 벤치마크
    └── 채택률 분석
```

### 5.4 Phase 4: Option A - Disaggregated (1.5주)

```
Week 5-6:
├── Day 1-3: KV Cache 전송 인프라
│   ├── KVCacheSender / KVCacheReceiver
│   ├── TCP / RDMA / SHM 백엔드
│   └── 비동기 전송
│
├── Day 4-6: PrefillNode / DecodeNode 구현
│   ├── Prefill 최적화 모델 로드
│   ├── Decode + MoE Offload + N-gram 통합
│   └── 조율자 (Coordinator)
│
└── Day 7-10: 멀티노드 테스트
    ├── 단일 머신 테스트
    └── 멀티 머신 테스트 (선택적)
```

### 5.5 Phase 5: Option B - APEX (1주)

```
Week 7:
├── Day 1-2: APEXProfiler 구현
│   ├── GPU 처리량 측정
│   ├── CPU 처리량 측정
│   └── 최적 비율 계산
│
├── Day 3-4: APEXScheduler + APEXExecutor 구현
│   ├── 배치 분할 로직
│   ├── GPU/CPU 병렬 실행
│   └── 결과 병합
│
└── Day 5-7: 테스트 및 튜닝
    ├── Dense 모델 (Llama) 테스트
    └── 다양한 GPU 환경 테스트
```

### 5.6 Phase 6: 통합 및 최적화 (1주)

```
Week 8:
├── Day 1-3: 전체 통합
│   ├── Option A/B 스위칭 로직
│   ├── 에러 처리
│   └── 로깅 및 모니터링
│
├── Day 4-5: 성능 최적화
│   ├── 병목 분석
│   ├── 메모리 최적화
│   └── 지연시간 최적화
│
└── Day 6-7: 문서화 및 릴리스
    ├── 사용자 가이드
    ├── API 문서
    └── 벤치마크 결과
```

---

## 6. 파일 구조

```
vllm/
├── config.py                          # [수정] HybridConfig 추가
├── entrypoints/
│   └── openai/
│       └── cli_args.py                # [수정] 하이브리드 인자 추가
│
├── executor/
│   ├── hybrid_executor.py             # [신규] 하이브리드 실행기 팩토리
│   ├── apex_executor.py               # [신규] Option B: APEX
│   └── disaggregated_executor.py      # [신규] Option A: Disaggregated
│
├── model_executor/
│   └── layers/
│       └── moe/
│           ├── expert_offload.py      # [신규] Expert 오프로드 관리자
│           └── fused_moe_offload.py   # [신규] 오프로드 MoE 레이어
│
├── spec_decode/
│   ├── ngram_proposer.py              # [신규] N-gram 제안자
│   └── ngram_spec_worker.py           # [신규] N-gram Spec 워커
│
├── engine/
│   └── disaggregated/
│       ├── prefill_node.py            # [신규] Prefill 노드
│       ├── decode_node.py             # [신규] Decode 노드
│       ├── coordinator.py             # [신규] 조율자
│       └── kv_transfer.py             # [신규] KV 캐시 전송
│
├── worker/
│   └── cpu_worker_int8.py             # [신규] INT8 CPU 워커
│
└── platforms/
    └── intel_cpu_utils.py             # [수정] AVX-512 VNNI 최적화 추가

csrc/cpu/
├── gemm_vnni.cpp                      # [신규] VNNI GEMM (이전 계획에서)
├── expert_compute.cpp                 # [신규] Expert CPU 계산 커널
└── torch_bindings.cpp                 # [수정] 바인딩 추가

tests/
├── hybrid/
│   ├── test_moe_offload.py            # [신규]
│   ├── test_ngram_proposer.py         # [신규]
│   ├── test_disaggregated.py          # [신규]
│   └── test_apex.py                   # [신규]
└── benchmarks/
    └── bench_hybrid_options.py        # [신규]

docs/
├── HYBRID_OPTIONS_IMPLEMENTATION_PLAN.md  # 이 문서
└── HYBRID_USER_GUIDE.md               # [신규] 사용자 가이드
```

---

## 7. API 설계

### 7.1 Python API

```python
from vllm import LLM, HybridConfig, ExpertOffloadConfig, NGramSpecConfig

# Option A: MoE 최적화
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    hybrid_config=HybridConfig(
        mode="option-a",
        moe_offload=ExpertOffloadConfig(
            enabled=True,
            num_gpu_experts=8,
        ),
        ngram_spec=NGramSpecConfig(
            enabled=True,
            n=3,
            num_speculative_tokens=5,
        ),
    ),
    tensor_parallel_size=8,
)

# Option B: APEX
llm = LLM(
    model="meta-llama/Llama-3-70B",
    hybrid_config=HybridConfig(
        mode="option-b",
        apex=APEXConfig(
            enabled=True,
            auto_profile=True,
        ),
    ),
    tensor_parallel_size=8,
)

# 추론
outputs = llm.generate(prompts, sampling_params)
```

### 7.2 REST API 확장

```
# 상태 확인
GET /v1/hybrid/status
Response:
{
  "mode": "option-a",
  "components": {
    "moe_offload": {
      "enabled": true,
      "gpu_experts": 8,
      "cpu_experts": 248
    },
    "ngram_spec": {
      "enabled": true,
      "acceptance_rate": 0.72
    },
    "disaggregated": {
      "enabled": true,
      "prefill_nodes": 1,
      "decode_nodes": 1
    }
  }
}

# 런타임 설정 변경 (선택적)
POST /v1/hybrid/config
{
  "ngram_num_speculative_tokens": 7
}
```

---

## 8. 테스트 계획

### 8.1 단위 테스트

```python
# tests/hybrid/test_moe_offload.py

class TestExpertOffloadManager:

    def test_expert_routing(self):
        """Expert 라우팅 정확성"""
        manager = ExpertOffloadManager(num_experts=256, num_gpu_experts=8)

        hidden = torch.randn(32, 4096)
        router_logits = torch.randn(32, 256)

        output = manager.route_and_compute(hidden, router_logits)

        assert output.shape == hidden.shape

    def test_cpu_expert_int8(self):
        """CPU Expert INT8 계산 정확성"""
        expert = MoEExpert(4096, 11008)
        expert_int8 = quantize_expert_to_int8(expert)

        input_tensor = torch.randn(32, 4096)

        output_fp32 = expert(input_tensor)
        output_int8 = expert_int8.int8_forward(input_tensor)

        # 상대 오차 < 1%
        rel_error = (output_fp32 - output_int8).abs() / output_fp32.abs().clamp(min=1e-5)
        assert rel_error.mean() < 0.01

    def test_expert_swap(self):
        """Expert LRU 교체"""
        manager = ExpertOffloadManager(num_experts=16, num_gpu_experts=4)

        # Expert 0-3을 많이 사용
        for _ in range(100):
            manager.expert_usage_count[0] += 1
            manager.expert_usage_count[1] += 1

        manager._swap_experts()

        # 0, 1이 GPU에 있어야 함
        assert 0 in manager.gpu_experts
        assert 1 in manager.gpu_experts


# tests/hybrid/test_ngram_proposer.py

class TestNGramProposer:

    def test_ngram_update(self):
        """N-gram 테이블 업데이트"""
        proposer = NGramProposer(n=3)

        # "The quick brown fox" 학습
        tokens = [100, 200, 300, 400]  # The, quick, brown, fox
        proposer.update(tokens)

        # (quick, brown) → fox 매핑 확인
        assert (200, 300) in proposer.ngram_store
        assert 400 in proposer.ngram_store[(200, 300)]

    def test_ngram_propose(self):
        """N-gram 추측"""
        proposer = NGramProposer(n=3, min_frequency=1)

        # 학습
        proposer.update([100, 200, 300, 400, 500])
        proposer.update([100, 200, 300, 400, 500])  # 빈도 2

        # 추측
        proposals, _ = proposer.propose([100, 200, 300])

        assert proposals[0] == 400  # fox


# tests/hybrid/test_apex.py

class TestAPEXScheduler:

    def test_batch_partition(self):
        """배치 분할 테스트"""
        scheduler = APEXScheduler(gpu_ratio=0.8, cpu_ratio=0.2)

        requests = [InferenceRequest(id=i) for i in range(10)]

        gpu_batch, cpu_batch = scheduler.partition_batch(requests)

        assert len(gpu_batch) == 8
        assert len(cpu_batch) == 2
```

### 8.2 통합 테스트

```python
# tests/hybrid/test_integration.py

class TestOptionAIntegration:

    @pytest.mark.skipif(not is_moe_model_available(), reason="MoE 모델 필요")
    def test_full_option_a(self):
        """Option A 전체 통합 테스트"""

        llm = LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            hybrid_config=HybridConfig(
                mode="option-a",
                moe_offload=ExpertOffloadConfig(enabled=True),
                ngram_spec=NGramSpecConfig(enabled=True),
            ),
            tensor_parallel_size=8,
        )

        outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=50))

        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].text) > 0


class TestOptionBIntegration:

    def test_full_option_b(self):
        """Option B 전체 통합 테스트"""

        llm = LLM(
            model="meta-llama/Llama-3-8B",  # 작은 모델로 테스트
            hybrid_config=HybridConfig(
                mode="option-b",
                apex=APEXConfig(enabled=True),
            ),
        )

        outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=50))

        assert len(outputs) == 1
```

### 8.3 벤치마크

```python
# tests/benchmarks/bench_hybrid_options.py

def benchmark_all_modes():
    """모든 모드 벤치마크"""

    results = {}

    # GPU only (기준)
    results["gpu_only"] = benchmark_mode(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        hybrid_mode="none",
    )

    # Option A: Full
    results["option_a_full"] = benchmark_mode(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        hybrid_mode="option-a",
        moe_offload=True,
        ngram_spec=True,
    )

    # Option A: MoE only
    results["option_a_moe"] = benchmark_mode(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        hybrid_mode="option-a",
        moe_offload=True,
        ngram_spec=False,
    )

    # Option B
    results["option_b"] = benchmark_mode(
        model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        hybrid_mode="option-b",
    )

    # 결과 출력
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)

    for mode, result in results.items():
        print(f"\n{mode}:")
        print(f"  Throughput: {result['throughput']:.2f} tok/s")
        print(f"  TTFT: {result['ttft']:.2f} ms")
        print(f"  TPOT: {result['tpot']:.2f} ms")
        print(f"  vs GPU only: {result['throughput']/results['gpu_only']['throughput']:.2f}x")
```

---

## 9. 예상 성능

### 9.1 Option A (DeepSeek R1 70B)

| 구성 | 처리량 (tok/s) | vs GPU only |
|------|----------------|-------------|
| GPU only | 5,742 | 1.0x |
| + MoE Offload | 7,000-8,000 | 1.2-1.4x |
| + N-gram Spec | 8,500-10,000 | 1.5-1.7x |
| + Disaggregated | 10,000-15,000 | 1.7-2.6x |

### 9.2 Option B (parallel-batch) 예상 성능

실측 수치는 `docs/paper/main.tex` 의 평가 섹션과 `eval/results/` 하위 raw 로그를
참조. 특정 기종 (H100 / T4 등) 에 고정된 tok/s 숫자는 런타임 자동 감지 원칙과 상충하므로
삭제했다.

---

## 10. 결론

### 현재 옵션 선택 가이드

| 조건 | 권장 옵션 |
|------|----------|
| Dense 모델 + CPU 여유 있음 | **parallel-batch** (구현 완료) |
| MoE 모델 (DeepSeek, Mixtral) | moe-hybrid (미래 구현) |
| CPU 미탑재 또는 NUMA 없음 | GPU only (`--hybrid-mode none`) |

### 핵심 성공 요인 (parallel-batch)

1. **Dual process + CapacityAwareRouter**: CPU-first routing + overflow to GPU → GPU non-interference
2. **per-NUMA `cpu_max_num_seqs = 1` 고정**: 1 시퀀스가 NUMA 의 모든 물리 코어를 OMP 병렬로 점유
3. **3 차원 fallback chain**: attention kernel (custom_avx → ipex → sdpa_batched → sdpa_loop) / thread bind (C++ _C_utils → Python) / ISA (AMX → AVX-512 VNNI → AVX-512 → AVX2)

---

*마지막 업데이트: 2026-04-11*
*버전: 2.0 (CapacityAwareRouter / CUDA 13.0 / torch 2.9 반영)*
