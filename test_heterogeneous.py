import torch
import vllm
print(f"DEBUG: vllm imported from {vllm.__file__}")
from vllm import LLM, SamplingParams

def test_heterogeneous_generation():
    # 현재 시스템의 GPU 개수 확인
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")

    # 테스트를 위해 TP=2로 고정 (1개의 GPU가 있는 환경에서 1 GPU + 1 CPU 테스트)
    tensor_parallel_size = 2
    print(f"Testing with tensor_parallel_size={tensor_parallel_size} (Heterogeneous mode)")

    # Heterogeneous device 설정으로 LLM 초기화
    try:
        llm = LLM(
            model="facebook/opt-125m",  # 가벼운 모델 사용
            tensor_parallel_size=tensor_parallel_size,
            device="heterogeneous",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        # 혹시 메모리 부족 등으로 실패할 경우를 대비해 예외 처리
        return

    # 테스트 프롬프트 생성
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    
    # 샘플링 파라미터 설정
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)

    # 추론 실행
    outputs = llm.generate(prompts, sampling_params)

    # 결과 출력
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    test_heterogeneous_generation()
