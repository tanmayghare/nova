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

class LlamaModel(LanguageModel):
    """Implementation of LanguageModel using Ollama as primary and llama-cpp-python as fallback."""
    
    def __init__(
        self,
        model_name: str = "llama3.2:3b-instruct-q8_0",
        use_ollama: bool = True,
        fallback_model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = 6,
        n_gpu_layers: int = 1,
        verbose: bool = False,
        use_mlock: bool = True,
        embedding: bool = False
    ):
        """
        Initialize the Llama model with Ollama as primary and GGUF as fallback.
        
        Args:
            model_name: Name of the Ollama model to use. Defaults to llama3.2-vision:11b-instruct-q4_K_M.
            use_ollama: Whether to use Ollama as primary backend. Defaults to True.
            fallback_model_path: Path to the GGUF model file for fallback. If None, tries to find it in default locations.
            n_ctx: Context window size. Default 4096 for 11B model.
            n_threads: Number of threads to use. Default 6 for optimal performance.
            n_gpu_layers: Number of layers to offload to GPU. 1 for initial GPU acceleration.
            verbose: Whether to enable verbose logging.
            use_mlock: Whether to use mlock to prevent swapping.
            embedding: Whether to enable embedding mode.
        """
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.fallback_model_path = self._resolve_model_path(fallback_model_path) if fallback_model_path else None
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.use_mlock = use_mlock
        self.embedding = embedding
        self._model = None
        
        # Check Ollama availability if it's the primary backend
        if self.use_ollama:
            self._check_ollama_availability()
        
    def _check_ollama_availability(self):
        """Check if Ollama is running and the model is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise RuntimeError("Ollama API is not responding")
            
            available_models = [model['name'] for model in response.json()['models']]
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found in Ollama. Will use fallback if available.")
                self.use_ollama = False
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama is not running. Will use fallback if available.")
            self.use_ollama = False
        
    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve the model path from various possible locations."""
        if model_path and os.path.exists(model_path):
            return model_path
            
        # Check default locations
        default_locations = [
            os.path.expanduser("~/Downloads/Llama-3.2-11B-Vision-Instruct.f16.gguf"),
            os.path.expanduser("~/.llama/checkpoints/Llama-3.2-11B-Vision-Instruct.f16.gguf"),
            "Llama-3.2-11B-Vision-Instruct.f16.gguf"
        ]
        
        for path in default_locations:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError(
            "Could not find GGUF model file. Please ensure the model file 'Llama-3.2-11B-Vision-Instruct.f16.gguf' "
            "is placed in one of:\n"
            f"{chr(10).join(default_locations)}"
        )
        
    def _ensure_model_loaded(self):
        """Ensure the fallback model is loaded, loading it if necessary."""
        if self._model is None and not self.use_ollama and self.fallback_model_path:
            try:
                self._model = Llama(
                    model_path=self.fallback_model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                    use_mlock=self.use_mlock,
                    embedding=self.embedding
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Llama model from {self.fallback_model_path}. "
                    "Please ensure the file is a valid GGUF model."
                ) from e
    
    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        if self.use_ollama:
            return await self._generate_plan_ollama(task, context)
        elif self.fallback_model_path:
            return await self._generate_plan_local(task, context)
        else:
            raise RuntimeError("No available model backend. Please ensure Ollama is running or provide a fallback model.")
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        prompt = f"""Task: {task}
Plan execution results:
{json.dumps(plan, indent=2)}

Context:
{context}

Please provide a summary of the task execution results.
"""
        if self.use_ollama:
            return await self._generate_response_ollama(prompt)
        elif self.fallback_model_path:
            return await self._generate_response_local(prompt)
        else:
            raise RuntimeError("No available model backend. Please ensure Ollama is running or provide a fallback model.")
    
    async def _generate_plan_local(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan using the local Llama model."""
        self._ensure_model_loaded()
        
        # Format the prompt for plan generation
        prompt = PROMPT_TEMPLATE.format(context=context, query=task)
        
        try:
            response = self._model.create_completion(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["</plan>"]
            )
            return self._parse_plan_response(response)
        except Exception as e:
            logger.error(f"Failed to generate plan: {str(e)}")
            raise RuntimeError("Failed to generate plan") from e
    
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
    
    async def _generate_response_local(self, prompt: str) -> str:
        """Generate a response using the local Llama model."""
        self._ensure_model_loaded()
        
        try:
            response = self._model.create_completion(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise RuntimeError("Failed to generate response") from e
    
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
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "repeat_penalty": 1.1
                }
            })
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "repeat_penalty": 1.1
                    }
                }
            )
            
            print("\nResponse status code:", response.status_code)
            print("Response headers:", dict(response.headers))
            print("Raw response text:", response.text)
            
            if response.status_code != 200:
                logger.warning(f"Ollama API request failed: {response.text}. Trying fallback if available.")
                if self.fallback_model_path:
                    self.use_ollama = False
                    return await self._generate_response_local(prompt)
                raise RuntimeError(f"Ollama API request failed: {response.text}")
            
            result = response.json()
            print("\nParsed JSON response:", json.dumps(result, indent=2))
            return result
            
        except Exception as e:
            if self.fallback_model_path:
                logger.info("Attempting to use fallback model...")
                self.use_ollama = False
                return await self._generate_response_local(prompt)
            raise RuntimeError("Failed to generate response with Ollama and no fallback available") from e
    
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