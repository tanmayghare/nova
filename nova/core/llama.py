"""Llama language model integration."""

import os
from typing import Any, Dict, List, Optional, cast

from llama_cpp import Llama  # type: ignore


class LlamaModel:
    """Llama language model implementation.
    
    This class provides an interface to the Llama language model using llama.cpp
    for optimized performance on Apple Silicon Macs.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 1,
        n_threads: Optional[int] = None,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the Llama model.
        
        Args:
            model_path: Path to the quantized model file
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU
            n_threads: Number of threads to use (default: CPU cores)
            temperature: Sampling temperature
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            temperature=temperature,
        )
        
        self.system_prompt = """You are a helpful AI assistant that helps with web automation tasks.
You should generate plans and responses in a structured format.
Always be precise and follow the given context."""

    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task.
        
        Args:
            task: The task to execute
            context: Additional context for the task
            
        Returns:
            List of plan steps
        """
        prompt = f"""System: {self.system_prompt}

Context: {context}

Task: {task}

Generate a step-by-step plan to complete this task. Each step should be either a tool usage or a browser action.
Format the response as a JSON list of steps, where each step has:
- type: "tool" or "browser"
- tool: tool name (for tool steps)
- action: action details (for browser steps)
- input: input parameters (for tool steps)

Response:"""

        response = self.model(
            prompt,
            max_tokens=1024,
            stop=["</response>"],
            temperature=0.3,  # Lower temperature for more structured output
        )
        
        # TODO: Parse the response into structured plan
        # For now, return a mock plan
        return [
            {"type": "tool", "tool": "mock", "input": {"test": "data"}},
            {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}},
        ]

    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results.
        
        Args:
            task: The original task
            plan: The executed plan
            context: Additional context
            
        Returns:
            Generated response
        """
        prompt = f"""System: {self.system_prompt}

Context: {context}

Task: {task}

Plan executed: {plan}

Generate a response summarizing the results of the task execution.
Focus on the key outcomes and any important observations.

Response:"""

        response = self.model(
            prompt,
            max_tokens=1024,
            stop=["</response>"],
            temperature=0.7,
        )
        
        return cast(str, response["choices"][0]["text"].strip()) 