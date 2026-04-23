

# Researching CUDA Plugin Code

- [x] Identify CUDA/GPU related code in `/csrc` <!-- id: 0 -->
- [x] Determine how the C++ code is exposed to Python (module name) <!-- id: 1 -->
- [x] Find where the extension is imported in `vllm` <!-- id: 2 -->
- [x] List usages of the imported extension in `vllm` <!-- id: 3 -->
- [x] Summarize findings for the user <!-- id: 4 -->

# Explaining Build and Linkage Mechanism

- [x] Find and read `CMakeLists.txt` to see source file compilation list <!-- id: 5 -->
- [x] Read `csrc/ops.h` (or linked headers) to see function declarations <!-- id: 6 -->
- [x] Explain the linking process (CMake + Header files) to the user <!-- id: 7 -->

# Analyzing GPUModelRunner.execute_model

- [x] Locate `GPUModelRunner` and `execute_model` <!-- id: 8 -->
- [x] Read `execute_model` implementation <!-- id: 9 -->
- [x] Trace `model.forward` or equivalent calls <!-- id: 10 -->
- [x] Trace `sampler` or output processing calls <!-- id: 11 -->
- [x] Document the call graph and logic flow <!-- id: 12 -->

# Analyzing Mixtral Model

- [x] Read `vllm/model_executor/models/mixtral.py` <!-- id: 13 -->
- [x] Trace `MixtralMoE` and its kernel calls <!-- id: 14 -->
- [x] Document the MoE routing flow <!-- id: 15 -->

# Heterogeneous Platform Support Analysis

- [x] Analyze `vllm/platforms` for platform interface and registration <!-- id: 16 -->
- [x] Analyze `vllm/worker` for Worker/ModelRunner instantiation logic <!-- id: 17 -->
- [x] Analyze `vllm/distributed` for rank and device management <!-- id: 18 -->
- [x] Identify necessary changes for Heterogeneous (CPU+GPU) support <!-- id: 19 -->

# Heterogeneous Platform Implementation Plan

- [x] Create `vllm/platforms/heterogeneous.py` wrapper class <!-- id: 20 -->
- [x] Modify `vllm/worker/worker.py` to support heterogeneous device initialization <!-- id: 21 -->
- [x] Modify `vllm/distributed/parallel_state.py` for mixed-device groups <!-- id: 22 -->
- [x] Update `vllm/engine/arg_utils.py` to allow heterogeneous device selection <!-- id: 23 -->
- [x] Verify heterogeneous platform initialization with manual test <!-- id: 24 -->

# NUMA-aware Heterogeneous Optimization

- [x] Investigate NUMA node detection and process pinning (affinity) <!-- id: 25 -->
- [x] Update `HeterogeneousPlatform.device_count` to return GPUs + 2 <!-- id: 26 -->
- [x] Modify `vllm/worker/worker.py` to pin CPU workers to specific NUMA nodes <!-- id: 27 -->
- [x] Verify NUMA pinning logic (mock/simulation) <!-- id: 28 -->

