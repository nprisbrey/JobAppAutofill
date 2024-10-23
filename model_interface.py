from abc import ABC, abstractmethod
import json
import requests
from typing import Optional


class ModelInterface(ABC):
    """Abstract base class for model interfaces"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text response from the model"""
        pass

    @abstractmethod
    def stream_generate(self, prompt: str, callback) -> str:
        """Generate text response from the model with streaming"""
        pass


class OllamaInterface(ModelInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/generate"
        data = {"model": self.model_name, "prompt": prompt}

        try:
            with requests.post(url, json=data, stream=True) as response:
                response.raise_for_status()
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if "response" in json_response:
                            full_response += json_response["response"]
                return full_response
        except requests.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return "Error querying Ollama"

    def stream_generate(self, prompt: str, callback) -> str:
        url = f"{self.base_url}/generate"
        data = {"model": self.model_name, "prompt": prompt}

        try:
            with requests.post(url, json=data, stream=True) as response:
                response.raise_for_status()
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if "response" in json_response:
                            chunk = json_response["response"]
                            full_response += chunk
                            callback(chunk)
                return full_response
        except requests.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return "Error querying Ollama"


class HuggingFaceInterface(ModelInterface):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Import here to avoid loading dependencies unless HF interface is used
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading model from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        print("Now starting generation...")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=32,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        print("Finished generation")
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[
            len(prompt) :
        ]

    def stream_generate(self, prompt: str, callback) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        full_response = ""

        print("Now starting generation...")
        for outputs in self.model.generate(
            **inputs,
            max_new_tokens=32,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streaming=True,
        ):
            print("Doing another token")
            token = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            full_response += token
            callback(token)
        print("Finished generation")

        return full_response[len(prompt) :]


def get_model_interface(model_type: str, model_name: str) -> Optional[ModelInterface]:
    """Factory function to create appropriate model interface"""
    if model_type.lower() == "ollama":
        return OllamaInterface(model_name)
    elif model_type.lower() == "huggingface":
        return HuggingFaceInterface(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
