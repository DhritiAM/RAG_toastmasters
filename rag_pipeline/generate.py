import os
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Install with: pip install openai")

import requests
import json


class LLMGenerator:
    """
    LLM Generator that supports both OpenAI API and Ollama (local) models.
    Automatically detects which to use based on model name and configuration.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize LLM generator.
        
        Args:
            model_name: Model name. For OpenAI: "gpt-3.5-turbo", "gpt-4", etc.
                       For Ollama: "phi3", "llama2", etc.
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI API (for OpenAI-compatible APIs)
        """
        self.model = model_name
        self.use_openai = self._is_openai_model(model_name)
        
        if self.use_openai:
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                )
            # Initialize OpenAI client
            # API key should come from environment variable for security
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
                    "Example (Windows): set OPENAI_API_KEY=your-api-key-here\n"
                    "Example (Linux/Mac): export OPENAI_API_KEY=your-api-key-here"
                )
            
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            
            self.client = OpenAI(**client_kwargs)
            if base_url:
                print(f"Using OpenAI-compatible API at: {base_url}")
            else:
                print(f"Using OpenAI API with model: {model_name}")
        else:
            # Use Ollama (local)
            self.url = "http://localhost:11434/api/generate"
            print(f"Using Ollama (local) with model: {model_name}")
    
    def _is_openai_model(self, model_name: str) -> bool:
        """Check if model name indicates OpenAI model"""
        openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
        return any(model_name.startswith(prefix) for prefix in ["gpt-", "o1-", "o3-"]) or model_name in openai_models
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate (None = model default)
        
        Returns:
            Generated text response
        """
        if self.use_openai:
            return self._generate_openai(prompt, temperature, max_tokens)
        else:
            return self._generate_ollama(prompt)
    
    def _generate_openai(self, prompt: str, temperature: float, max_tokens: Optional[int]) -> str:
        """Generate using OpenAI API"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {e}")
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama API (local)"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["response"]
        except Exception as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")


# Backward compatibility alias
LocalLLMGenerator = LLMGenerator
