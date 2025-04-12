"""Vision capabilities for Nova."""

from typing import Any, Dict

from pydantic import BaseModel, Field

from .llama import LlamaModel


class VisionConfig(BaseModel):
    """Configuration for vision capabilities."""
    model_name: str = Field(
        default="mistral-small3.1:24b-instruct-2503-q4_K_M",
        description="Name of the Ollama model to use"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for sampling"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate"
    )


class VisionModel:
    """Vision model implementation using Llama."""

    def __init__(
        self,
        model_name: str = "mistral-small3.1:24b-instruct-2503-q4_K_M",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize the vision model.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
        """
        self.config = VisionConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._model = LlamaModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        **kwargs: Any
    ) -> str:
        """Analyze an image using the vision model.
        
        Args:
            image_path: Path to the image
            prompt: Prompt for image analysis
            **kwargs: Additional arguments
            
        Returns:
            Analysis result
        """
        return await self._model.generate_with_vision(
            prompt=prompt,
            image_path=image_path,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return self.config.dict() 