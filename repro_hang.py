
import asyncio
import os
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, LoadConfig, DecodingConfig, ObservabilityConfig, CompilationConfig
from vllm.v1.engine.async_llm import AsyncLLM
import vllm
print(f"DEBUG_AG: vllm imported from {vllm.__file__}")

# Mocking the configuration to match user's setup
async def main():
    print("Starting reproduction script...")
    
    # 1. emulate "device='heterogeneous'"
    model_config = ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=0,
    )
    
    # Parallel Config
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=2,
        worker_use_ray=False,
        max_parallel_loading_workers=None,
        distributed_executor_backend="mp",
    )
    
    # Device Config - CRITICAL
    device_config = DeviceConfig(device="heterogeneous")
    
    # Cache Config
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.6,
        swap_space=4,
        cache_dtype="auto",
    )
    
    # Scheduler Config
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        max_model_len=2048,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=LoadConfig(),
        decoding_config=DecodingConfig(),
        observability_config=ObservabilityConfig(),
    )
    
    # Override compilation config to match logs but shorter
    vllm_config.compilation_config.cudagraph_capture_sizes = [16, 1]
    
    print(f"Config created. Device type: {vllm_config.device_config.device_type}")
    
    try:
        print("Initializing AsyncLLM...")
        llm = await AsyncLLM.from_vllm_config(vllm_config)
        print("AsyncLLM initialized successfully!")
    except Exception as e:
        print(f"AsyncLLM initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure VLLM_LOGGING_LEVEL is DEBUG
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    asyncio.run(main())
