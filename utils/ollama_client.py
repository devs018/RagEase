import ollama
import json
from typing import Dict, Any, List, Optional
from config import Config

class OllamaClient:
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or Config.OLLAMA_MODEL
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.client = ollama.Client(host=self.base_url)
        
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send chat messages to Ollama and get response
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response['message']['content']
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response for a single prompt
        """
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                **kwargs
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def is_model_available(self) -> bool:
        """
        Check if the model is available in Ollama
        """
        try:
            models = self.client.list()
            model_names = [m["model"] for m in models.get("models", [])]
            return self.model in model_names
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False
    
    def pull_model_if_needed(self) -> bool:
        """
        Pull the model if it's not available
        """
        if not self.is_model_available():
            try:
                print(f"Pulling model {self.model}...")
                self.client.pull(self.model)
                print(f"Model {self.model} pulled successfully!")
                return True
            except Exception as e:
                print(f"Error pulling model: {e}")
                return False
        return True

    def create_system_message(self, content: str) -> Dict[str, str]:
        return {"role": "system", "content": content}
    
    def create_user_message(self, content: str) -> Dict[str, str]:
        return {"role": "user", "content": content}
    
    def create_assistant_message(self, content: str) -> Dict[str, str]:
        return {"role": "assistant", "content": content}