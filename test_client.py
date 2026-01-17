import requests
import json

def test_generate():
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "facebook/opt-125m",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        text = result['choices'][0]['text']
        print(f"Success! Output: {text}")
    except Exception as e:
        print(f"Error: {e}")
        if 'response' in locals():
            print(f"Response text: {response.text}")

if __name__ == "__main__":
    test_generate()
