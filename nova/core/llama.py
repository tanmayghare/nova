"""Llama language model integration."""

import os
import logging
import re
import json
from typing import Any, Dict, List, Optional, cast, Union
from pathlib import Path

from llama_cpp import Llama  # type: ignore
from nova.core.monitoring import PerformanceMonitor
from ..core.language_model import LanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a global performance monitor instance
monitor = PerformanceMonitor()

PROMPT_TEMPLATE = """You are an AI assistant that generates structured plans in valid JSON format.
The plan should be a single JSON object with a "steps" array containing action objects.
Each action object must have "action", "target", and "value" fields.
Do not include any text outside the JSON object.

Context: {context}

User: {query}
"""

class LlamaModel(LanguageModel):
    """Implementation of LanguageModel using llama-cpp-python."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False
    ):
        """
        Initialize the Llama model.
        
        Args:
            model_path: Path to the GGUF model file. If None, tries to find it in default locations.
            n_ctx: Context window size.
            n_threads: Number of threads to use. If None, uses system default.
            n_gpu_layers: Number of layers to offload to GPU. 0 for CPU only.
            verbose: Whether to enable verbose logging.
        """
        self.model_path = self._resolve_model_path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._model = None
        
    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        """Resolve the model path from various possible locations."""
        if model_path and os.path.exists(model_path):
            return model_path
            
        # Check default locations
        default_locations = [
            os.path.expanduser("~/.llama/checkpoints/Llama3.2-3B/llama-3.2-3b-instruct.Q4_K_M.gguf"),
            os.path.expanduser("~/.llama/checkpoints/llama-3.2-3b-instruct.Q4_K_M.gguf"),
            "llama-3.2-3b-instruct.Q4_K_M.gguf"
        ]
        
        for path in default_locations:
            if os.path.exists(path):
                return path
                
        raise FileNotFoundError(
            "Could not find GGUF model file. Please download it from "
            "https://huggingface.co/hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF and place it in one of:\n"
            f"{chr(10).join(default_locations)}"
        )
        
    def _ensure_model_loaded(self):
        """Ensure the model is loaded, loading it if necessary."""
        if self._model is None:
            try:
                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Llama model from {self.model_path}. "
                    "Please ensure the file is a valid GGUF model."
                ) from e
    
    @monitor.track("plan_generation")
    def generate_plan(self, task: str, context: Dict[str, str]) -> Dict[str, str]:
        """Generate a plan for executing a task.
        
        Args:
            task: The task to execute
            context: Additional context for the task
            
        Returns:
            A structured plan as a dictionary
        """
        self._ensure_model_loaded()
        
        # Format the prompt for plan generation
        prompt = PROMPT_TEMPLATE.format(context=context.get('current_state', 'No context available'), query=task)
        
        try:
            response = self._model(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                stop=["</plan>"]
            )
            return self._parse_plan_response(response)
        except Exception as e:
            logger.error(f"Failed to generate plan: {str(e)}")
            raise RuntimeError("Failed to generate plan") from e
    
    @monitor.track("response_generation")
    def generate_response(self, prompt: str) -> str:
        """Generate a response to a prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated response text
        """
        self._ensure_model_loaded()
        
        try:
            response = self._model(
                prompt,
                max_tokens=1024,
                temperature=0.7
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise RuntimeError("Failed to generate response") from e
            
    def _parse_plan_response(self, response: Dict) -> Dict[str, str]:
        """Parse the model's response into a structured plan."""
        try:
            # Extract the text from the response
            text = response['choices'][0]['text'].strip()
            
            # Clean up any potential artifacts
            text = text.replace('<|eot_id|>', '')
            text = text.replace('<|end_of_text|>', '')
            
            # Find all JSON objects in the response
            json_objects = []
            start = 0
            while True:
                # Find the next JSON object
                start = text.find('{', start)
                if start == -1:
                    break
                    
                # Track nested braces to find the matching closing brace
                brace_count = 1
                pos = start + 1
                while brace_count > 0 and pos < len(text):
                    if text[pos] == '{':
                        brace_count += 1
                    elif text[pos] == '}':
                        brace_count -= 1
                    pos += 1
                
                if brace_count == 0:
                    # Found a complete JSON object
                    json_str = text[start:pos]
                    try:
                        obj = json.loads(json_str)
                        json_objects.append(obj)
                    except json.JSONDecodeError:
                        pass  # Skip invalid JSON objects
                    start = pos
                else:
                    break  # Incomplete JSON object
            
            if not json_objects:
                raise ValueError("No valid JSON objects found in response")
            
            # Use the last complete JSON object
            plan = json_objects[-1]
            
            # Validate the structure
            if not isinstance(plan, dict):
                raise ValueError("Plan must be a JSON object")
            
            if "steps" not in plan:
                raise ValueError("Plan must contain a 'steps' array")
            
            if not isinstance(plan["steps"], list):
                raise ValueError("'steps' must be an array")
            
            # Validate each step
            for step in plan["steps"]:
                if not isinstance(step, dict):
                    raise ValueError("Each step must be an object")
                
                if "action" not in step or "target" not in step or "value" not in step:
                    raise ValueError("Each step must have 'action', 'target', and 'value' fields")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to parse plan response: {str(e)}")
            logger.error(f"Raw response: {text}")
            raise ValueError("Failed to parse plan response") from e

    def _parse_response(self, response: str) -> dict:
        """Parse the model's response into a structured plan."""
        try:
            # Clean up the response
            response = response.strip()
            
            # Remove control tokens
            response = re.sub(r'<\|.*?\|>', '', response)
            
            # Find the first valid JSON object
            json_start = response.find('{')
            if json_start == -1:
                raise ValueError("No JSON object found in response")
            
            json_end = response.find('}', json_start) + 1
            if json_end == 0:
                raise ValueError("Incomplete JSON object")
            
            # Extract the first JSON object
            json_str = response[json_start:json_end]
            
            # Parse the JSON
            plan = json.loads(json_str)
            
            # Validate the structure
            if not isinstance(plan, dict):
                raise ValueError("Plan must be a JSON object")
            
            if "steps" not in plan:
                raise ValueError("Plan must contain a 'steps' array")
            
            if not isinstance(plan["steps"], list):
                raise ValueError("'steps' must be an array")
            
            # Validate each step
            for step in plan["steps"]:
                if not isinstance(step, dict):
                    raise ValueError("Each step must be an object")
                
                if "action" not in step or "target" not in step or "value" not in step:
                    raise ValueError("Each step must have 'action', 'target', and 'value' fields")
                
            return plan
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse plan response: {str(e)}")
            self.logger.error(f"Raw response: {response}")
            raise ValueError(f"Failed to parse plan response: {str(e)}")
        except ValueError as e:
            self.logger.error(f"Invalid plan structure: {str(e)}")
            self.logger.error(f"Raw response: {response}")
            raise ValueError(f"Invalid plan structure: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing plan: {str(e)}")
            self.logger.error(f"Raw response: {response}")
            raise ValueError(f"Unexpected error parsing plan: {str(e)}")