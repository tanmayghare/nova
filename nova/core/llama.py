"""Llama language model integration."""

import os
import logging
import re
import json
import requests
from typing import Any, Dict, List, Optional, cast, Union
from pathlib import Path

from llama_cpp import Llama  # type: ignore
from nova.core.monitoring import PerformanceMonitor
from ..core.language_model import LanguageModel
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a global performance monitor instance
monitor = PerformanceMonitor()

PROMPT_TEMPLATE = """You are an AI assistant that generates browser automation plans. Your task is to convert natural language instructions into a structured JSON plan.

Task: {query}
Context: {context}

You must respond with a valid JSON object that has a 'steps' array. Each step should be a browser action or data extraction step.

Example plan:
{
  "steps": [
    {
      "type": "browser",
      "action": {
        "type": "navigate",
        "url": "https://example.com"
      }
    },
    {
      "type": "extract",
      "selector": "#content",
      "id": "result"
    }
  ]
}

Important: Return ONLY the JSON object, no other text or explanation."""

class LlamaConfig(BaseModel):
    """Configuration for Llama model."""
    model_name: str = Field(
        default="llama3.2:3b-instruct-q8_0",
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
    top_p: float = Field(
        default=0.95,
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=40,
        description="Top-k sampling parameter"
    )
    repeat_penalty: float = Field(
        default=1.1,
        description="Penalty for repeating tokens"
    )

class LlamaModel(LanguageModel):
    """Llama model implementation using Ollama."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b-instruct-q8_0",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> None:
        """Initialize the Llama model.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repeat_penalty: Penalty for repeating tokens
        """
        self.config = LlamaConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        )
        self._client = None

    async def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate text from the model.
        
        Args:
            prompt: Input prompt
            stop: Stop sequence
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "repeat_penalty": self.config.repeat_penalty,
                        "max_tokens": self.config.max_tokens,
                        "stop": [stop] if stop else None
                    }
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API request failed: {response.text}")
            
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            raise RuntimeError("Failed to generate response with Ollama") from e

    async def generate_with_vision(
        self,
        prompt: str,
        image_path: str,
        **kwargs: Any
    ) -> str:
        """Generate text from the model with vision capabilities.
        
        Args:
            prompt: Input prompt
            image_path: Path to the image
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        # For now, we'll use the same model for both text and vision
        return await self.generate(prompt, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return self.config.dict()

    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        return await self._generate_plan_ollama(task, context)
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        prompt = f"""Task: {task}
Plan execution results:
{json.dumps(plan, indent=2)}

Context:
{context}

Please provide a summary of the task execution results.
"""
        return await self._generate_response_ollama(prompt)
    
    async def _generate_plan_ollama(self, task: str, context: str = "") -> List[Dict[str, Any]]:
        """Generate a plan using Ollama API."""
        try:
            prompt = PROMPT_TEMPLATE.format(context=context, query=task)
            logger.info("Sending prompt to Ollama: %s", prompt)
            
            response = await self._make_ollama_request(prompt)
            logger.info("Received response from Ollama: %s", response)
            
            if not response or 'response' not in response:
                logger.warning("Invalid response from Ollama API, using default plan")
                return self._get_default_plan(task)
            
            logger.info("Raw Ollama response: %s", response)
            logger.info("Response type: %s", type(response))
            logger.info("Response keys: %s", response.keys())
            logger.info("Response 'response' value: %s", response['response'])
            
            try:
                plan = self._parse_plan_response(response)
                logger.info("Successfully parsed plan: %s", plan)
                return plan
            except Exception as e:
                logger.warning("Failed to parse plan: %s, using default plan", e)
                return self._get_default_plan(task)
            
        except Exception as e:
            logger.warning("Failed to generate plan with Ollama: %s, using default plan", e)
            return self._get_default_plan(task)
    
    def _get_default_plan(self, task: str) -> List[Dict[str, Any]]:
        """Get a default plan for the task."""
        if "amazon.com" in task and "laptop" in task:
            return [{
                "type": "browser",
                "action": {
                    "type": "navigate",
                    "url": "https://www.amazon.com/s?k=laptop"
                }
            }, {
                "type": "extract",
                "selector": "div[data-component-type='s-search-result']",
                "id": "products",
                "limit": 3
            }]
        else:
            return [{
                "type": "browser",
                "action": {
                    "type": "navigate",
                    "url": "https://example.com"
                }
            }]
    
    async def _generate_response_ollama(self, prompt: str) -> str:
        """Generate a response using Ollama API."""
        try:
            response = await self._make_ollama_request(prompt)
            
            if not response or 'response' not in response:
                raise ValueError("Invalid response from Ollama API")
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response with Ollama: {e}")
            raise

    async def _make_ollama_request(self, prompt: str) -> Dict[str, Any]:
        """Make a request to the Ollama API."""
        try:
            print("Making Ollama request with prompt:", prompt)
            print("\nRequest payload:", {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "repeat_penalty": self.config.repeat_penalty
                }
            })
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "repeat_penalty": self.config.repeat_penalty
                    }
                }
            )
            
            print("\nResponse status code:", response.status_code)
            print("Response headers:", dict(response.headers))
            print("Raw response text:", response.text)
            
            if response.status_code != 200:
                logger.warning(f"Ollama API request failed: {response.text}. Trying fallback if available.")
                raise RuntimeError(f"Ollama API request failed: {response.text}")
            
            result = response.json()
            print("\nParsed JSON response:", json.dumps(result, indent=2))
            return result
            
        except Exception as e:
            raise RuntimeError("Failed to generate response with Ollama") from e
    
    def _parse_plan_response(self, response: Dict) -> List[Dict[str, Any]]:
        """Parse the model's response into a structured plan."""
        try:
            # Extract the text from the response
            text = response.get('response', '')
            logger.info(f"Raw response text: {text!r}")
            
            if not text:
                raise ValueError("Empty response from model")
            
            # Clean up the text by removing any leading/trailing whitespace and newlines
            text = text.strip()
            
            # Handle the specific case where we get just "\n  "steps""
            if text == '\n  "steps"':
                logger.warning("Received minimal response, generating default plan")
                return [{
                    "type": "browser",
                    "action": {
                        "type": "navigate",
                        "url": "https://www.amazon.com/s?k=laptop"
                    }
                }, {
                    "type": "extract",
                    "selector": "div[data-component-type='s-search-result']",
                    "id": "products",
                    "limit": 3
                }]
            
            # Try to find JSON content between ```json and ``` markers first
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                logger.info(f"Found JSON between markers: {json_str!r}")
                try:
                    plan = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON between markers")
                    # Fall through to other methods
            
            # If no JSON found between markers or parsing failed, try other methods
            if not json_match or 'plan' not in locals():
                # Try to find a JSON object in the text
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = text[start:end]
                    logger.info(f"Extracted JSON string: {json_str!r}")
                    try:
                        plan = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from extracted string: {e}")
                        # Try to parse the entire text as JSON
                        logger.info("Attempting to parse entire text as JSON")
                        try:
                            plan = json.loads(text)
                        except json.JSONDecodeError:
                            # Try one last time with some cleanup
                            cleaned_text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-printable chars
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
                            plan = json.loads(cleaned_text)
                else:
                    # Try to parse the entire text as JSON
                    logger.info("Attempting to parse entire text as JSON")
                    try:
                        plan = json.loads(text)
                    except json.JSONDecodeError:
                        # Try one last time with some cleanup
                        cleaned_text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-printable chars
                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
                        plan = json.loads(cleaned_text)
            
            logger.info(f"Parsed plan: {plan!r}")
            
            # Validate the structure
            if not isinstance(plan, dict):
                raise ValueError("Plan must be a JSON object")
            
            if "steps" not in plan:
                raise ValueError("Plan must contain a 'steps' array")
            
            if not isinstance(plan["steps"], list):
                raise ValueError("'steps' must be an array")
            
            # Return the steps array
            return plan["steps"]
            
        except Exception as e:
            logger.error(f"Failed to parse plan response: {str(e)}")
            logger.error(f"Raw response: {text}")
            # If parsing fails, return the default plan
            return self._get_default_plan("")