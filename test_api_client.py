import requests
import json
import time

def test_remote_generation():
    # vLLM API 서버 주소 (기본값)
    api_url = "http://localhost:8000/v1/completions"
    
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "facebook/opt-125m",
        "prompt": "The capital of South Korea is",
        "max_tokens": 20,
        "temperature": 0.7
    }

    print(f"Sending request to {api_url}...")
    try:
        response = requests.post(api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            text = result['choices'][0]['text']
            print("\nSuccess!")
            print(f"Prompt: {data['prompt']}")
            print(f"Generated Text: {text}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Connection failed. Make sure the vLLM server is running.")
        print("Server launch command example:")
        print("python -m vllm.entrypoints.api_server --model facebook/opt-125m --device heterogeneous --tensor-parallel-size 4")

if __name__ == "__main__":
    test_remote_generation()
