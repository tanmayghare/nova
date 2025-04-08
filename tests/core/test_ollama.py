import requests
import json

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
            "model": "llama3.2:3b-instruct-q8_0",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.95,
                "repeat_penalty": 1.1
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