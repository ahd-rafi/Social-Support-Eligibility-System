"""LLM integration using Ollama."""

import logging
import json
from typing import Dict, Any, Optional, List

try:
    import ollama
except ImportError:
    ollama = None

from src.config import settings

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Wrapper for Ollama LLM interactions."""

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        """Initialize Ollama LLM.
        
        Args:
            model: Model name (default from settings)
            host: Ollama host URL (default from settings)
        """
        if ollama is None:
            raise ImportError(
                "ollama package is required. Install with: pip install ollama"
            )
        
        self.model = model or settings.ollama_model
        self.host = host or settings.ollama_host
        
        # Configure client
        self.client = ollama.Client(host=self.host)
        
        logger.info(f"Ollama LLM initialized: {self.model} @ {self.host}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
    ) -> str:
        """Generate text completion.
        
        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            format: Output format ("json" for JSON mode)
            
        Returns:
            Generated text
        """
        messages = []
        
        if system:
            messages.append({
                "role": "system",
                "content": system,
            })
        
        messages.append({
            "role": "user",
            "content": prompt,
        })
        
        options = {
            "temperature": temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        
        try:
            logger.debug(f"Generating with {self.model} (temp={temperature})")
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=options,
                format=format,
            )
            
            content = response['message']['content']
            
            logger.debug(f"Generated {len(content)} characters")
            
            return content
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate JSON output.
        
        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON as dict
        """
        response = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            format="json",
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """Multi-turn chat.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if Ollama server is available and model exists.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # List available models
            models = self.client.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            # Check if our model is available
            # Handle both "model:tag" and "model" formats
            model_base = self.model.split(':')[0]
            available = any(
                model_base in name or name.startswith(self.model)
                for name in model_names
            )
            
            if available:
                logger.info(f"✓ Model available: {self.model}")
            else:
                logger.warning(f"✗ Model not found: {self.model}")
                logger.info(f"  Available models: {model_names}")
            
            return available
            
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False


# Global instance
_llm = None


def get_llm() -> OllamaLLM:
    """Get or create global LLM instance."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM()
    return _llm


if __name__ == "__main__":
    # Test Ollama connection
    logging.basicConfig(level=logging.INFO)
    
    llm = OllamaLLM()
    
    if llm.is_available():
        print(f"✓ Ollama is available")
        print(f"  Model: {llm.model}")
        print(f"  Host: {llm.host}")
        
        # Test generation
        try:
            response = llm.generate("Say hello in one sentence.")
            print(f"\nTest generation:")
            print(f"  Response: {response}")
        except Exception as e:
            print(f"\n✗ Generation test failed: {e}")
    else:
        print(f"✗ Ollama is not available")
        print(f"  Make sure Ollama is running: ollama serve")
        print(f"  And model is pulled: ollama pull {llm.model}")
