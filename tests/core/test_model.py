from llama_cpp import Llama
import os
import requests
import json
from typing import Optional, Dict, Any

def test_llama_cpp():
    model_path = os.path.expanduser("~/Downloads/Llama-3.2-11B-Vision-Instruct.f16.gguf")
    print(f"Loading model from: {model_path}")
    print(f"File size: {os.path.getsize(model_path)} bytes")

    try:
        # Initialize with Metal support and latest configurations
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=1,
            verbose=True,
            use_mlock=True,
            embedding=False
        )
        print("Model loaded successfully!")
        
        # Try a simple completion with the new API
        output = model.create_completion(
            "Hello, how are you?",
            max_tokens=10,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1
        )
        print("\nTest completion:")
        print(output)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def test_ollama():
    """Test Ollama API integration"""
    try:
        # Test if Ollama is running
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            print("Ollama is running!")
            print("Available models:", response.json())
            
            # Try a simple completion
            completion_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2-vision",
                    "prompt": "Hello, how are you?",
                    "stream": False
                }
            )
            if completion_response.status_code == 200:
                print("\nTest completion:")
                print(completion_response.json())
        else:
            print("Ollama is not running or not accessible")
    except Exception as e:
        print(f"Error testing Ollama: {str(e)}")

if __name__ == "__main__":
    print("Testing llama-cpp-python...")
    test_llama_cpp()
    
    print("\nTesting Ollama...")
    test_ollama() 