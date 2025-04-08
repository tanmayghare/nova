"""Vision model integration for image analysis."""

import os
import logging
import base64
import requests
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

VISION_PROMPT_TEMPLATE = """You are an AI assistant with vision capabilities.
Analyze the following image and respond to the query.

Query: {query}

Please provide a detailed response based on what you see in the image.
"""

class VisionModel:
    """Handles vision-specific functionality for the Llama model."""
    
    def __init__(
        self,
        model_name: str = "llama3.2-vision",
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1
    ):
        """
        Initialize the vision model.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Controls randomness in the output
            top_p: Controls diversity via nucleus sampling
            repeat_penalty: Controls repetition in the output
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        
    def _encode_image(self, image_path: str) -> str:
        """Encode an image file to base64."""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            raise RuntimeError(f"Failed to encode image: {str(e)}") from e
        
    def _encode_image_from_url(self, image_url: str) -> str:
        """Download and encode an image from URL to base64."""
        try:
            response = requests.get(image_url)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download image from {image_url}")
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image from URL: {str(e)}")
            raise RuntimeError(f"Failed to encode image from URL: {str(e)}") from e
        
    def analyze_image(self, image_path: str, query: str) -> str:
        """Analyze an image using the vision model.
        
        Args:
            image_path: Path to the image file
            query: Question or instruction about the image
            
        Returns:
            Model's analysis of the image
        """
        try:
            # Encode image
            image_data = self._encode_image(image_path)
            
            # Format the prompt
            prompt = VISION_PROMPT_TEMPLATE.format(query=query)
            
            # Make API request
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "images": [image_data],
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "repeat_penalty": self.repeat_penalty
                    }
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API request failed: {response.text}")
            
            result = response.json()
            return result['response'].strip()
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {str(e)}")
            raise RuntimeError("Failed to analyze image") from e

    def analyze_image_url(self, image_url: str, query: str) -> str:
        """Analyze an image from a URL using the vision model.
        
        Args:
            image_url: URL of the image
            query: Question or instruction about the image
            
        Returns:
            Model's analysis of the image
        """
        try:
            # Encode image
            image_data = self._encode_image_from_url(image_url)
            
            # Format the prompt
            prompt = VISION_PROMPT_TEMPLATE.format(query=query)
            
            # Make API request
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "images": [image_data],
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "repeat_penalty": self.repeat_penalty
                    }
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API request failed: {response.text}")
            
            result = response.json()
            return result['response'].strip()
            
        except Exception as e:
            logger.error(f"Failed to analyze image from URL: {str(e)}")
            raise RuntimeError("Failed to analyze image from URL") from e 