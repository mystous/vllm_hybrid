# probe_map — qwen

| 구간 | 실제 경로 (파일:함수) | 계측 수단 | 상태 |
|---|---|---|---|
| S00 요청 수신·토큰화·스케줄 | sglang tokenizer_manager / scheduler (HTTP 프로세스, sglang::scheduler_TP*) | 요청별 TTFT (client) | 부분 (서버 내부 시각 미수집) |
| S01 attention·MoE 입력 | glm4_moe.py/qwen3_moe.py forward → triton/flashinfer attention 커널 | D2 커널명 | 이름 수준 |
| S02 router·top-k | sglang/srt/layers/moe/topk.py select_experts → fused_moe_triton layer.py:376 KTEPWrapperMethod.apply (kt_ep_wrapper.py:370-430) | D2 커널명 | 이름 수준 |
| S03 CPU 입력 전달 | kt_kernel/experts_base.py submit_forward: pinned slot ring (KExpertsCPUBuffer, slot=layer_idx%depth) .copy_(non_blocking) D2H | D2 gpu_memcpy | bytes 는 트레이스 args |
| S04 CPU 작업 전달·대기 | callback-free: cpuinfer.h go_on_stream (memop) → poll_loop_ (poller 스레드) → TaskQueue enqueue; task_queue.cpp KT_TQ_TIMING (wait_us) | D1 [kt-tq] | 집계만 |
| S05 CPU expert 계산 | operators/amx/moe_base.hpp forward_* → moe.hpp do_gate_up_gemm/do_down_gemm (qlen>MIN_QLEN or m>=MIN_ROWS → amx::mat_mul, 아니면 amx::vec_mul(AVX-512 VNNI)); KT_PHASE_PROF (prepare/cpy_input/q_input/up_gate/act/q_down/down/weight) | D1 cpu_jobs.csv | 64회당 1회 표본, 층 id 없음 |
| S06 CPU 완료·반환 | cpuinfer.h done 플래그 → experts_base.py sync_forward wait_done_on_stream → output_gpu.copy_(output_cpu, non_blocking) H2D | D2 gpu_memcpy | 부분 |
| S07 GPU hot expert | fused_moe_triton (kernel 이름 fused_moe_kernel 등) | D2 | 이름 수준 |
| S08 결합 전 의존 대기 | kt_ep_wrapper.py apply: output = gpu + cpu_output (sync) | D2 MoE 커널 사이 유휴 | PARTIAL_DEPENDENCY_MEASUREMENT |
| S09 결합·TP 통신·잔차 | all-reduce (custom allreduce 커널), residual add | D2 커널명 | 이름 수준 |
| S10 sampling·출력 | sampler 커널, detokenizer 프로세스 | D2 / client | 부분 |
| S11 graph 경로 | decode_cuda_graph_runner replay (capture bs 목록 runtime_proofs) | D2 (graph 내부 커널은 CUPTI 로 기록) | 이름 수준 |
| S12 동적 배치·layerwise prefill | 설치본 미지원 | — | NOT_APPLICABLE |
