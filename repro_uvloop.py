
import os
import asyncio
import uvloop
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.config import (VllmConfig, ModelConfig, CacheConfig, 
                        ParallelConfig, SchedulerConfig, DeviceConfig, 
                        LoadConfig, LoRAConfig, DecodingConfig, 
                        ObservabilityConfig, CompilationConfig)
from vllm.v1.core.sched.scheduler import Scheduler as V1Scheduler
from vllm.sampling_params import SamplingParams

# Install uvloop policy
uvloop.install()

async def main():
    print("Starting reproduction script with uvloop...")
    
    # 1. Create a VllmConfig object
    # Matches the one inferred from logs, but simplified construction
    
    vllm_config = VllmConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=0,
        skip_tokenizer_init=False,
        revision=None,
        max_seq_len_to_capture=2048,
        
        # Model Config
        model_config=ModelConfig(
            model="facebook/opt-125m",
            tokenizer="facebook/opt-125m",
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            seed=0,
            max_model_len=2048,
        ),
        
        # Parallel Config - Heterogeneous
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=2, # Force 2 workers (1 GPU + 1 CPU)
            worker_use_ray=False,
            distributed_executor_backend="mp", # Multiprocessing
        ),
        
        # Cache Config
        cache_config=CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.6,
            swap_space=4,
            cache_dtype="auto",
        ),
        
        # Scheduler Config
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=2048,
            max_num_seqs=256,
            max_model_len=2048,
            scheduler_cls=V1Scheduler,
        ),
        
        # Device Config
        device_config=DeviceConfig(
            device="heterogeneous",
        ),
        
        # Load Config
        load_config=LoadConfig(
            load_format="auto",
            download_dir=None,
            model_loader_extra_config=None,
        ),
        
        # Compilation Config
        compilation_config=CompilationConfig(),
        
        # LoRA Config
        lora_config=None,
        
        # Decoding Config
        decoding_config=DecodingConfig(),
        
        # Observability Config
        observability_config=ObservabilityConfig(),
    )
    
    # Override compilation config to match logs but shorter
    vllm_config.compilation_config.cudagraph_capture_sizes = [16, 1]
    
    print(f"Config created. Device type: {vllm_config.device_config.device_type}")
    
    # 2. Initialize AsyncLLM
    print("Initializing AsyncLLM...")
    try:
        llm = AsyncLLM.from_vllm_config(vllm_config)
        print("AsyncLLM initialized successfully!")
    except Exception as e:
        print(f"AsyncLLM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Generate
    print("Generating...")
    inputs = "Hello, my name is"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
    
    try:
        results_generator = llm.generate(inputs, sampling_params, request_id="req_0")
        async for request_output in results_generator:
            if request_output.finished:
                print(f"Generated: {request_output.outputs[0].text}")
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
