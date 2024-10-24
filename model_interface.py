from abc import ABC, abstractmethod
import ollama
from typing import Optional, Dict, Any
from selenium.webdriver.remote.webelement import WebElement
import torch
from torch.distributions.categorical import Categorical


class ModelInterface(ABC):
    """Abstract base class for model interfaces"""

    @abstractmethod
    def generate(self, prompt: str, generation_params: dict = None) -> str:
        """Generate text response from the model"""
        pass


class OllamaInterface(ModelInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client(host="http://localhost:11434")

    def generate(self, prompt: str, generation_params: dict = None) -> str:
        """
        Generate text using the Ollama model without streaming.
        Ignores generation_params as Ollama has its own configuration.
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model from {model_path}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )

    def _custom_generation_method(
        self, inputs, is_question: bool, field_info: Dict[str, Any]
    ) -> str:
        """
        Custom generation method that takes into account whether the field is a question.

        Args:
            inputs: The tokenized prompt
            is_question: Boolean indicating if the field label ends with a question mark
            field_info: Dictionary containing additional field information (type, label, etc.)

        Returns:
            str: Generated response
        """
        # If this is a question, than we are willing to have a higher
        # temperature and experiment more with the distrubtion, seeing
        # as there may not be one correct answer
        if is_question:
            temperature = 1.5
        # If this isn't a question (like "Last Name" or "School"), than
        # there may be one correct answer, meaning we want to drop the
        # temperature
        else:
            temperature = 0.9

        # Tensor of token ids of shape:
        # (batch_size, sequence_length)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_tok_len = input_ids.shape[-1]

        while True:
            print(f"input_ids: {input_ids.shape}")
            # Get CausalLMOutputWithPast object
            output_with_past = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # Get logits before softmax as torch.FloatTensor of shape:
            # (batch_size, sequence_length, vocab_size)
            all_logits = output_with_past.logits
            print(f"All logits shape: {all_logits.shape}")

            # Get rid of all logits except for last token, giving shape:
            # (batch_size, vocab_size)
            last_logits = all_logits[:, -1, :]
            print(f"Last logits shape: {last_logits.shape}")

            # Softmax with temperature, returning Tensor of shape:
            # (batch_size, vocab_size)
            probs = temp_softmax(last_logits, temperature)
            print(f"Probs shape: {probs.shape}")

            # Create distribution from probs and sample, giving tensor of shape:
            # (batch_size)
            dists = Categorical(probs)
            print(f"dists: {dists}")
            sampled_tok_idxs = dists.sample()
            print(f"sampled_tok_idxs: {sampled_tok_idxs}")

            # Append tensors
            input_ids = torch.cat((input_ids, sampled_tok_idxs.reshape(-1, 1)), dim=-1)
            print(f"New input_ids shape: {input_ids.shape}")

            # NOTE: We only check first sequence is done generating
            # If we're done generating
            if sampled_tok_idxs[0] == self.tokenizer.eos_token_id:
                return self.tokenizer.batch_decode(input_ids[:, prompt_tok_len:-1])[0]

    def generate(self, prompt: str, generation_params: dict = None) -> str:
        if generation_params is None:
            generation_params = {"method": "greedy"}

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Base generation parameters
        generate_kwargs = {
            "max_new_tokens": 32,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Configure generation method based on parameters
        method = generation_params.get("method", "greedy")
        match method:
            case "greedy":
                # Use default settings for greedy search
                generate_kwargs.update({"num_beams": 1})

            case "beam":
                # Configure beam search
                beam_size = generation_params.get("beam_size", 5)
                generate_kwargs.update(
                    {
                        "num_beams": beam_size,
                        "early_stopping": True,
                        "no_repeat_ngram_size": 2,
                    }
                )

            case "top_k":
                # Configure Top-K sampling
                top_k = generation_params.get("top_k", 50)
                generate_kwargs.update(
                    {"do_sample": True, "top_k": top_k, "temperature": 0.7}
                )

            case "top_p":
                # Configure Top-P (nucleus) sampling
                top_p = generation_params.get("top_p", 0.9)
                generate_kwargs.update(
                    {
                        "do_sample": True,
                        "top_p": top_p,
                        "top_k": 0,  # Disable top-k when using top-p
                        "temperature": 0.7,
                    }
                )

            case "custom":
                is_question = generation_params.get("is_question", False)
                field_info = generation_params.get("field_info", {})
                return self._custom_generation_method(inputs, is_question, field_info)

        outputs = self.model.generate(**inputs, **generate_kwargs)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[
            len(prompt) :
        ]
        return generated_text


def temp_softmax(logits, temp):
    """
    Args:
        logits: torch.Tensor in the shape of (batch_size, vocab_size)
        temp: A float of the temperature (0-1)

    Returns:
        torch.Tensor: Temperature softmaxed logits of shape (batch_size, vocab_size)
    """
    # Because we're changing the temperature, first divide all
    # logits by the temperature
    temp_logits = torch.div(logits, temp)

    # Calculate the sum of all scaled logits to the power of e
    e_powered = torch.exp(temp_logits)
    e_powered_sum = torch.sum(e_powered, dim=-1, keepdim=True)

    # Divide each original temperatured logit by the sum and return
    return torch.div(e_powered, e_powered_sum)


def get_model_interface(model_type: str, model_name: str) -> Optional[ModelInterface]:
    """Factory function to create appropriate model interface"""
    if model_type.lower() == "ollama":
        return OllamaInterface(model_name)
    elif model_type.lower() == "huggingface":
        return HuggingFaceInterface(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
