import ollama
from typing import List, Dict
from app.config import OLLAMA_MODEL, MAX_TOKENS, TEMPERATURE
from loguru import logger


class OllamaLLM:
    """
    Wrapper around the Ollama local inference server.

    Ollama runs as a background process (http://localhost:11434) and
    exposes a REST API. The Python client wraps that API.

    ollama==0.3.x API note:
      response = ollama.chat(...)
      response.message.content   ← object attribute (not response["message"]["content"])
      ollama.list().models        ← list of Model objects with .model attribute
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response given a conversation history.

        messages format (OpenAI-compatible):
          [
            {"role": "system",    "content": "You are..."},
            {"role": "user",      "content": "Context...\n\nQuestion: ..."},
          ]

        options:
          temperature  — 0.1: near-deterministic for factual Q&A
          num_predict  — max tokens to generate (not input tokens)
          top_p        — nucleus sampling: only consider tokens whose
                         cumulative probability exceeds 0.9.
                         Combined with low temperature, this prevents
                         rare-token artifacts while staying deterministic.
        """
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS,
                    "top_p": 0.9,
                }
            )
            # ollama client returns dicts when connected to Ollama server 0.6+
            # Support both dict and object response formats
            if isinstance(response, dict):
                return response["message"]["content"]
            return response.message.content
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return (
                "I encountered an error generating a response. "
                "Please ensure Ollama is running (`ollama serve` in a terminal)."
            )

    def is_available(self) -> bool:
        """
        Check if Ollama is running and the configured model is available.
        Used by the UI to show green/red status indicator.
        """
        try:
            r = ollama.list()
            # Support dict response (server 0.6+) and object response
            if isinstance(r, dict):
                models = r.get("models", [])
                model_names = [m.get("name", m.get("model", "")) for m in models]
            else:
                model_names = [m.model for m in r.models]
            base_name = OLLAMA_MODEL.split(":")[0]
            return any(base_name in name for name in model_names)
        except Exception:
            return False
