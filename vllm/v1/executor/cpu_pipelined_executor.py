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

import dataclasses
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

    def _get_pool_cpu_ids_str(self) -> Optional[str]:
        """driver_worker.local_omp_cpuid range string ("0-47" 등) 반환.

        `init_cpu_threads_env` (C++) 가 이 형식 그대로 받음. "all" 이거나
        비어있으면 None.
        """
        try:
            s = getattr(self.driver_worker, "local_omp_cpuid", None)
        except Exception:
            return None
        if isinstance(s, str) and s and s != "all":
            return s
        return None

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
            cpu_ids_str = self._get_pool_cpu_ids_str()

            def _initializer():
                # 1) pool thread 의 affinity 를 NUMA 전체로 일단 확장.
                #    main 이 pinned cpu_ids[0] 에 갇혀있고 pool thread 는 이를
                #    상속한 상태 — init_cpu_threads_env 가 parallel for 로
                #    워커들을 각 core 에 pin 하려면 그 core 들에 접근 가능해야.
                if cpu_set:
                    import os as _os
                    try:
                        _os.sched_setaffinity(0, cpu_set)
                    except Exception as e:
                        logger.warning(
                            "[HYBRID-CPU-EXEC-POOL] pool thread "
                            "sched_setaffinity failed: %s", e)

                # 2) C++ init_cpu_threads_env 를 pool thread 에서 호출해
                #    **pool thread 의 새 OMP team 을 1:1 pin**.
                #    sync 경로의 team 은 main thread 가 master 인 별개 team
                #    이므로 영향 없음. pool 이 master 인 team 만 이 호출로
                #    pin 됨 → sync 와 동일한 pinning.
                if cpu_ids_str:
                    try:
                        import torch as _t
                        _t.ops._C_utils.init_cpu_threads_env(cpu_ids_str)
                        logger.info(
                            "[HYBRID-CPU-EXEC-POOL] pool thread OMP team "
                            "1:1 pinned via init_cpu_threads_env(%r)",
                            cpu_ids_str)
                    except Exception as e:
                        logger.warning(
                            "[HYBRID-CPU-EXEC-POOL] init_cpu_threads_env "
                            "on pool thread failed: %s. OMP team 은 "
                            "unpinned (affinity 확장만 적용).", e)

            self._compute_pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cpu-compute",
                initializer=_initializer)
            logger.info(
                "[HYBRID-CPU-EXEC-POOL] Pipelined CPU executor ACTIVE "
                "(max_concurrent_batches=%d, compute pool max_workers=1, "
                "cpu_set_size=%s, cpu_ids=%r)",
                self.max_concurrent_batches,
                len(cpu_set) if cpu_set else "None",
                cpu_ids_str)

    @property
    def max_concurrent_batches(self) -> int:
        """EngineCore 에서 batch_queue 를 활성화하는 조건 (> 1)."""
        return 2

    @staticmethod
    def _snapshot_output(out):
        """ModelRunnerOutput 의 input_batch 공유 mutable ref 를 snapshot.

        GPUModelRunner.execute_model 은 `ModelRunnerOutput(req_ids=self.
        input_batch.req_ids, req_id_to_index=self.input_batch.req_id_to_index,
        ...)` 로 **input_batch 의 mutable list/dict 를 reference** 로 반환한다
        (v1/worker/gpu_model_runner.py:1763-1764).

        Sync 에서는 execute_model 직후 scheduler.update_from_output 이 호출돼서
        다음 step 의 _update_states 가 input_batch 를 수정하기 전에 소비되므로
        문제 없음.

        Async (batch_queue) 에서는 step N 의 ModelRunnerOutput 이 batch_queue
        에 머무는 동안 step N+1 의 _update_states 가 input_batch 를 덮어쓰면
        step N 의 req_id_to_index 도 **동일 dict 로** 함께 변경되어 다음 race
        가 발생:
            main: batch_queue pop 해서 update_from_output 호출
            → result_N.req_id_to_index 는 이미 N+1 용으로 수정됨
            → KeyError: 원래 req_id 가 dict 에 없음 → EngineDeadError.

        Fix: pool 에서 forward 결과 반환 전에 req_ids / req_id_to_index 만
        shallow copy 로 스냅샷. 나머지 필드 (sampled_token_ids, logprobs 등)
        는 매 step 새로 생성되는 local 데이터라 공유가 없음.
        """
        if out is None:
            return out
        return dataclasses.replace(
            out,
            req_ids=list(out.req_ids),
            req_id_to_index=dict(out.req_id_to_index),
        )

    def _run_and_snapshot(self, scheduler_output):
        # super(PipelinedCPUExecutor, self).execute_model 는 MRO 상
        # Executor.execute_model → collective_rpc("execute_model") →
        # WorkerWrapperBase → CPUWorker.execute_model → ModelRunnerOutput.
        out = super(
            PipelinedCPUExecutor, self).execute_model(scheduler_output)
        return self._snapshot_output(out)

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
        # _run_and_snapshot 이 CPUWorker.execute_model 호출 후 결과의 mutable
        # reference 를 snapshot — step N+1 이 input_batch 를 수정해도 pending
        # 된 step N 의 ModelRunnerOutput 이 오염되지 않도록.
        future: Future = self._compute_pool.submit(
            self._run_and_snapshot, scheduler_output)
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
