# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pipelined CPU Executor — X Phase 3 구현

핵심 아이디어:
    UniProcExecutor 를 상속하고 `max_concurrent_batches > 1` 로 선언하면,
    EngineCore 의 `step_with_batch_queue` 경로가 자동 활성화된다 (core.py
    line 137, 509). 이 경로는 이미 vllm 이 pipeline parallelism 용으로 만든
    것인데, 우리는 CPU engine 의 pipeline 을 위해 재사용.

    execute_model() 은 Future 를 반환해야 함. 이를 위해 ThreadPoolExecutor
    로 compute 를 다른 thread 에서 돌리고, future 자체를 반환.

    EngineCore.step_with_batch_queue 가 알아서:
      1. scheduler.schedule(N)
      2. future_N = executor.execute_model(scheduler_output_N)   # 즉시 리턴
      3. batch_queue.put((future_N, scheduler_output_N))
      4. 다음 iter 에서 scheduler.schedule(N+1)  ← compute(N) 병행 중
      5. ...

효과:
    step(N+1) 의 scheduler.schedule() 과 step(N) 의 model.forward() 가
    **같은 프로세스의 다른 thread 에서 병렬** — torch CPU matmul 이 GIL
    해제하는 동안 main Python thread 가 scheduler 를 돌림.

core.py 는 수정하지 않음 (CLAUDE.md 원칙 유지). 기존 pipeline parallelism
infrastructure 를 CPU async pipeline 에 재활용.

활성화: HYBRID_CPU_ASYNC_EXECUTOR=1 env var.
기본 off → 기존 sync 경로 유지 (A/B 비교 가능).
"""
from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

from vllm.logger import init_logger
from vllm.v1.executor.abstract import UniProcExecutor
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


def is_async_executor_enabled() -> bool:
    """Feature flag — `HYBRID_CPU_ASYNC_EXECUTOR=1` 이면 활성."""
    return os.environ.get("HYBRID_CPU_ASYNC_EXECUTOR", "0") == "1"


class PipelinedCPUExecutor(UniProcExecutor):
    """CPU 용 pipelined executor.

    - `max_concurrent_batches = 2` → EngineCore 가 `step_with_batch_queue`
      사용 (pipeline 자동 활성).
    - `execute_model` 은 Future 반환 (ThreadPoolExecutor 에 submit).
    - ThreadPoolExecutor(max_workers=1) — 한 engine 당 하나의 compute
      thread. 이 thread 가 OMP master 역할.

    Main Python thread 는 future 반환 즉시 EngineCore 로 복귀 → 다음
    step 의 scheduler.schedule() 병행 실행 가능.
    """

    _compute_pool: Optional[ThreadPoolExecutor] = None

    def _init_compute_pool(self) -> None:
        """Lazy 초기화. Subclass `__init__` 호출 순서 민감하지 않도록."""
        if self._compute_pool is None:
            self._compute_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cpu-compute")
            logger.info(
                "[HYBRID-CPU-EXEC-POOL] Pipelined CPU executor ACTIVE "
                "(max_concurrent_batches=%d, compute pool max_workers=1)",
                self.max_concurrent_batches)

    @property
    def max_concurrent_batches(self) -> int:
        """EngineCore 에서 batch_queue 를 활성화하는 조건 (> 1)."""
        return 2

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """Future 를 반환. EngineCore 의 batch_queue 가 future.result() 함.

        Sync 인터페이스도 유지 — max_concurrent_batches=1 인 환경에서도
        호출 가능하지만 그 경우는 우리가 쓸 일 없음.
        """
        self._init_compute_pool()
        assert self._compute_pool is not None
        # UniProcExecutor.execute_model 은 collective_rpc 로 CPUWorker 호출
        future: Future = self._compute_pool.submit(
            super().execute_model, scheduler_output)
        return future

    def shutdown(self) -> None:
        """Shutdown compute pool then delegate to parent."""
        pool = self._compute_pool
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._compute_pool = None
        # UniProcExecutor.shutdown exists via base class
        try:
            super().shutdown()
        except AttributeError:
            pass
