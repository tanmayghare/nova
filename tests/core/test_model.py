import requests
import json
from typing import Optional, Dict, Any

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
                    "model": "llama3.2:3b-instruct-q8_0",
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
    print("\nTesting Ollama...")
    test_ollama() 