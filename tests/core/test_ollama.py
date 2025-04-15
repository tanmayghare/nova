import requests
import json
import os

def test_ollama():
    prompt = """You must respond with a valid JSON object that has a 'steps' array. No other text, just the JSON:

{
  "steps": [
    {
      "type": "browser",
      "action": {
        "type": "navigate",
        "url": "https://example.com"
      }
    }
  ]
}"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": os.environ.get("MODEL_NAME"),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": os.environ.get("MODEL_TEMPERATURE"),
                "top_p": os.environ.get("MODEL_TOP_P"),
                "repeat_penalty": os.environ.get("MODEL_REPETITION_PENALTY")
            }
        }
    )
    
    print("Response status:", response.status_code)
    print("Raw response:", response.text)
    
    if response.status_code == 200:
        result = response.json()
        print("\nParsed response:", json.dumps(result, indent=2))
        print("\nResponse type:", type(result))
        print("Response keys:", result.keys())
        if 'response' in result:
            print("\nResponse content:", result['response'])
            try:
                plan = json.loads(result['response'])
                print("\nParsed plan:", json.dumps(plan, indent=2))
            except json.JSONDecodeError as e:
                print("\nFailed to parse plan:", e)

if __name__ == "__main__":
    test_ollama() 