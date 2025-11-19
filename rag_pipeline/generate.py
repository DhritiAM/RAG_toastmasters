import requests
import json

class LocalLLMGenerator:
    def __init__(self, model_name="phi3"):
        self.model = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()

        return data["response"]
