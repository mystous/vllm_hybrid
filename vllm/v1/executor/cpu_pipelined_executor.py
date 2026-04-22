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

    def _get_pool_cpu_set(self) -> Optional[set]:
        """Pool thread 에 적용할 CPU affinity set 계산.

        init_cpu_threads_env (C++) 가 `#pragma omp parallel for schedule(static,1)`
        로 각 OMP 워커를 1:1 pin 할 때 OMP master (= main thread) 본인도
        `sched_setaffinity(0, ...)` 로 cpu_ids[0] 한 개에 pin 된다.

        이후 ThreadPoolExecutor 가 새 pool thread 를 pthread_create 로 생성하면
        creator (main) 의 affinity 를 상속 → pool thread 도 1 core 에 pinned.
        Pool thread 가 torch.mm 으로 OMP parallel region 에 진입하면 libgomp 는
        이를 새 master 로 간주하고 새 team 을 생성 — 새 OMP 워커들이 pool thread
        의 affinity 를 상속해서 **전체 팀이 1 core 에 집중** 된다.

        Sync 경로는 main 이 같은 master 로 기존 team 을 재사용하므로 워커들이
        이미 1:1 pin 된 상태가 유지되어 문제 없음. Async 는 master 가 바뀌면서
        이 이점이 깨짐.

        Fix: pool thread 의 affinity 를 driver_worker.local_omp_cpuid 로 명시된
        NUMA 노드 전체 core set 으로 확장. pool thread 가 만드는 OMP 워커들은
        그 확장된 affinity 를 상속 → NUMA 전체 core 에 spread.

        `driver_worker.local_omp_cpuid` 는 CPUWorker.init_device 에서 설정한
        range string (예: "0-47"). `all` 이면 None 반환 (pinning 없음 상태).
        """
        try:
            s = getattr(self.driver_worker, "local_omp_cpuid", None)
        except Exception:
            return None
        if not s or not isinstance(s, str) or s == "all":
            return None
        result: set = set()
        for chunk in s.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                try:
                    a, b = chunk.split("-", 1)
                    result.update(range(int(a), int(b) + 1))
                except ValueError:
                    continue
            else:
                try:
                    result.add(int(chunk))
                except ValueError:
                    continue
        return result or None

    def _init_compute_pool(self) -> None:
        """Lazy 초기화. Subclass `__init__` 호출 순서 민감하지 않도록."""
        if self._compute_pool is None:
            cpu_set = self._get_pool_cpu_set()

            def _initializer():
                if cpu_set:
                    import os as _os
                    try:
                        _os.sched_setaffinity(0, cpu_set)
                        logger.info(
                            "[HYBRID-CPU-EXEC-POOL] pool thread affinity "
                            "확장 → %d cores (OMP team 이 NUMA 전체 core "
                            "에 spread)", len(cpu_set))
                    except Exception as e:
                        logger.warning(
                            "[HYBRID-CPU-EXEC-POOL] pool thread "
                            "sched_setaffinity failed: %s. OMP team 이 main "
                            "의 pinned core 에 갇힐 수 있음.", e)

            self._compute_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cpu-compute",
                initializer=_initializer)
            logger.info(
                "[HYBRID-CPU-EXEC-POOL] Pipelined CPU executor ACTIVE "
                "(max_concurrent_batches=%d, compute pool max_workers=1, "
                "cpu_set_size=%s)",
                self.max_concurrent_batches,
                len(cpu_set) if cpu_set else "None")

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
