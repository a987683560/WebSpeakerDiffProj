import requests
from config_reader import ConfigReader


class OllamaClient:
    def __init__(self, url, model, prompt, temperature=0.7, top_p=0.9):
        self.url = url
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.top_p = top_p

    def generate_response(self, msg):
        data = {
            "model": self.model,
            "prompt": (self.prompt+msg),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()['response']
        except requests.RequestException as e:
            print(f"请求出错: {e}")
            return ''
config_reader = ConfigReader('config.json')
ollama_settings = config_reader.get_ollama_settings()
ollama = OllamaClient(
    ollama_settings.get('url'),
    ollama_settings.get('model'),
    ollama_settings.get('prompt'),
    ollama_settings.get('temperature', 0.7),
    ollama_settings.get('top_p', 0.9)
)
print(ollama.generate_response('自作自受'))
