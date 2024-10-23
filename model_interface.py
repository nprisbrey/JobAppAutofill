from abc import ABC, abstractmethod
import ollama
from typing import Optional
from selenium.webdriver.remote.webelement import WebElement


class ModelInterface(ABC):
    """Abstract base class for model interfaces"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text response from the model"""
        pass


class OllamaInterface(ModelInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client(host="http://localhost:11434")

    def generate(self, prompt: str) -> str:
        """
        Generate text using the Ollama model without streaming.
        """
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt)
            return response["response"]
        except ollama.ResponseError as e:
            print(f"Error generating with Ollama: {e.error}")
            if e.status_code == 404:
                print(f"Model {self.model_name} not found. Attempting to pull...")
                ollama.pull(self.model_name)
                return self.generate(prompt)
            return f"Error generating with Ollama: {e.error}"
        except Exception as e:
            print(f"Unexpected error in generate: {e}")
            return f"Unexpected error in generate: {e}"


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
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[
            len(prompt) :
        ]
        print(f"Finished generation: {generated_text}")
        # Process the complete response
        return generated_text


def get_model_interface(model_type: str, model_name: str) -> Optional[ModelInterface]:
    """Factory function to create appropriate model interface"""
    if model_type.lower() == "ollama":
        return OllamaInterface(model_name)
    elif model_type.lower() == "huggingface":
        return HuggingFaceInterface(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
