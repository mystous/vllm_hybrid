# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# SUB_201 §5 / B3 PoC — launch overhead telemetry.
# Qwen-7B suffix nsys 측정에서 cudaLaunchKernel 36.3% + cuLaunchKernelEx
# 20.2% + cudaGraphLaunch 17.8% = launch overhead 합 ~74%. async scheduler
# 의 한 iter 당 schedule 결정 횟수 / 평균 scheduled token 수를 측정해
# (num_iters_per_sec * tokens_per_iter) → expected launches / sec 와
# 실측 launch rate (18,606 calls/sec) 의 비교 가능. 측정만 하고 결정 path
# 자체는 변경 안 함 (PoC scope).
#
# Activation: ``VLLM_SCHED_LAUNCH_TELEMETRY=1`` 시 매 iter 의 counter
# 증가 + ``VLLM_SCHED_LAUNCH_TELEMETRY_LOG_EVERY`` (default 100) iter 당
# 한 번 logger.info 로 dump.
_LAUNCH_TELEMETRY_ENABLED: bool = (
    os.environ.get("VLLM_SCHED_LAUNCH_TELEMETRY") == "1"
)
_LAUNCH_TELEMETRY_LOG_EVERY: int = int(
    os.environ.get("VLLM_SCHED_LAUNCH_TELEMETRY_LOG_EVERY", "100")
)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        # SUB_201 §5 / B3 PoC — telemetry counters. Lazily ignored when
        # ``_LAUNCH_TELEMETRY_ENABLED`` is False (one extra attribute lookup
        # per iter, ~ns scale, no measurable hot-path overhead).
        self._b3_iter_count: int = 0
        self._b3_total_scheduled_tokens: int = 0
        self._b3_total_scheduled_reqs: int = 0
        self._b3_total_spec_reqs: int = 0

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        if _LAUNCH_TELEMETRY_ENABLED:
            self._b3_iter_count += 1
            self._b3_total_scheduled_tokens += (
                scheduler_output.total_num_scheduled_tokens
            )
            self._b3_total_scheduled_reqs += len(
                scheduler_output.num_scheduled_tokens
            )
            self._b3_total_spec_reqs += len(spec_decode_tokens)
            if (
                self._b3_iter_count > 0
                and self._b3_iter_count % _LAUNCH_TELEMETRY_LOG_EVERY == 0
            ):
                avg_tok = (
                    self._b3_total_scheduled_tokens / self._b3_iter_count
                )
                avg_req = self._b3_total_scheduled_reqs / self._b3_iter_count
                avg_spec = self._b3_total_spec_reqs / self._b3_iter_count
                logger.info(
                    "[SUB_201/B3] iters=%d avg_tokens/iter=%.2f "
                    "avg_reqs/iter=%.2f avg_spec_reqs/iter=%.2f "
                    "num_spec_tokens=%d",
                    self._b3_iter_count,
                    avg_tok,
                    avg_req,
                    avg_spec,
                    self.num_spec_tokens,
                )
        for req_id in scheduler_output.num_scheduled_tokens:
            # IDE_006 / TSK_015.B-3.b parallel — NEO swap_out 발화 후
            # cdec_req 가 self.requests 에서 제거됐는데
            # num_scheduled_tokens dict 에는 남는 영역 → KeyError 회피.
            request = self.requests.get(req_id)
            if request is None:
                continue
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
