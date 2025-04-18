from __future__ import annotations

"""Language model integration."""

import os
import logging
import json
import re

from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, field_validator

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from .schemas import (
    PlanGenerationOutput,
    ActionOutput,
    RecoveryPlanOutput,
    CommandInterpretationOutput
)

logger = logging.getLogger(__name__)


# --- Define Pydantic Schema for Plan Generation ---
class PlanStep(BaseModel):
    """A single step in the execution plan."""
    tool: str = Field(..., description="The name of the tool to use (e.g., 'navigate', 'click', 'type', 'finish').")
    input: Dict[str, Any] = Field(default_factory=dict, description="The input parameters for the tool.")

    @field_validator('tool')
    def tool_must_be_valid(cls, v):
        valid_tools = {'navigate', 'click', 'type', 'wait', 'screenshot', 'finish'}
        if v not in valid_tools:
            raise ValueError(f"Invalid tool name: {v}. Must be one of {valid_tools}")
        return v

    @field_validator('input')
    def validate_input_parameters(cls, v, info):
        tool = info.data.get('tool')
        if tool == 'navigate':
            if 'url' not in v:
                raise ValueError("URL is required for navigate tool")
            if not v['url'].startswith(('http://', 'https://')):
                raise ValueError("URL must start with http:// or https://")
        elif tool == 'click':
            if 'selector' not in v:
                raise ValueError("Selector is required for click tool")
        elif tool == 'type':
            if 'selector' not in v:
                raise ValueError("Selector is required for type tool")
            if 'text' not in v:
                raise ValueError("Text is required for type tool")
        return v

class PlanGenerationOutput(BaseModel):
    """The output format for plan generation."""
    plan: List[PlanStep] = Field(..., description="The list of steps in the execution plan.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The confidence score for the plan.")
    thought_process: str = Field(..., description="The reasoning behind the plan.")

    @field_validator('plan')
    def plan_must_contain_one_step(cls, v):
        if not v:
            raise ValueError("Plan must contain at least one step")
        if len(v) > 1:
            raise ValueError("Plan must contain exactly one step")
        return v

    @field_validator('confidence')
    def confidence_must_be_valid(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

# --- End Pydantic Schema ---


class LanguageModel(Protocol):
    """Protocol for language models."""
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan (including thought, confidence) for executing a task."""
        ...
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        ...

    async def interpret_command(self, command: str) -> Dict[str, Any]:
        """Interpret a natural language command into a structured plan.
        
        Args:
            command: The natural language command to interpret
            
        Returns:
            A dictionary containing:
            - goal: The identified overall goal
            - entities: Extracted entities (URLs, search terms, etc.)
            - plan: A sequence of tool actions to execute
        """
        ...


class LangChainAdapter:
    """Adapter for LangChain models to implement the LanguageModel protocol."""
    
    def __init__(self, model: BaseChatModel):
        """Initialize with a LangChain model."""
        self.model = model
        # Initialize output parsers
        self.plan_parser = PydanticOutputParser(pydantic_object=PlanGenerationOutput)
        self.action_parser = PydanticOutputParser(pydantic_object=ActionOutput)
        self.recovery_parser = PydanticOutputParser(pydantic_object=RecoveryPlanOutput)
        self.command_parser = PydanticOutputParser(pydantic_object=CommandInterpretationOutput)
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan for executing a task using structured output."""
        try:
            # Create the prompt with format instructions
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a task execution agent that generates a single next step for completing a task.
                
                CRITICAL RULES:
                1. You must generate EXACTLY ONE step in the plan
                2. For the 'navigate' tool, you MUST include the 'url' parameter in the input object
                3. For the 'type' tool, you MUST include 'selector', 'text', and 'submit' parameters
                4. For the 'click' tool, you MUST include the 'selector' parameter
                5. The confidence score must be between 0.0 and 1.0
                6. Check the current URL before deciding to navigate - only navigate if we're not already on the target page
                7. After navigation, use 'get_html' to get the page content before deciding next steps
                8. NEVER navigate to the same URL twice in a row - this is strictly forbidden
                9. If the current URL matches the target URL, you MUST use a different tool (like 'get_html' or 'get_text')
                10. If a navigation loop is detected in the history, you MUST use a different tool
                
                {format_instructions}
                
                Context: {context}"""),
                ("human", "{task}")
            ])
            
            # Format the prompt with the parser's instructions
            formatted_prompt = prompt.format_messages(
                format_instructions=self.plan_parser.get_format_instructions(),
                context=context,
                task=task
            )
            
            # Get the model's response
            response = await self.model.ainvoke(formatted_prompt)
            
            # Parse the response using the PydanticOutputParser
            parsed_output = self.plan_parser.parse(response.content)
            
            # Extract data from the validated Pydantic object
            thought = parsed_output.thought_process
            confidence = parsed_output.confidence
            plan_steps_dict = [step.model_dump() for step in parsed_output.plan]
            
            # Log success
            logger.info(f"Successfully generated structured plan: confidence={confidence:.2f}, steps={len(plan_steps_dict)}")
            logger.debug(f"Generated thought: {thought[:200]}...")
            logger.debug(f"Generated plan steps: {plan_steps_dict}")

            return thought, confidence, plan_steps_dict

        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            return "", 0.0, []
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        try:
            # Create a simple chat prompt without structured output
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are Nova, an AI assistant. Summarize the task execution based on the provided history."),
                ("human", f"Task: {task}\n\nExecution History:\n```\n{plan}\n```\n\nFinal Context:\n```\n{context}\n```\n\nBased on the execution history, provide a concise summary of what was done and the final outcome or result.")
            ])
            
            # Get the model's response
            response = await self.model.ainvoke(prompt.format_messages())
            return response.content
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            return f"Error generating response: {e}"
    
    async def interpret_command(self, command: str) -> Dict[str, Any]:
        """Interpret a command using structured output."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a command interpreter that extracts goals and entities from user commands.
                
                {format_instructions}
                """),
                ("human", "{command}")
            ])
            
            formatted_prompt = prompt.format_messages(
                format_instructions=self.command_parser.get_format_instructions(),
                command=command
            )
            
            response = await self.model.ainvoke(formatted_prompt)
            parsed_output = self.command_parser.parse(response.content)
            return parsed_output.model_dump()
            
        except Exception as e:
            logger.error(f"Command interpretation failed: {e}", exc_info=True)
            return {"goal": command, "entities": {}, "plan": []}
    
    async def generate_recovery_plan(self, error: str, action: Dict[str, Any], dom_context: str) -> Dict[str, Any]:
        """Generate a recovery plan using structured output."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an error recovery agent that suggests recovery steps when actions fail.
                
                {format_instructions}
                
                Previous action: {action}
                Error: {error}
                DOM context: {dom_context}
                """),
                ("human", "Suggest a recovery step.")
            ])
            
            formatted_prompt = prompt.format_messages(
                format_instructions=self.recovery_parser.get_format_instructions(),
                action=str(action),
                error=error,
                dom_context=dom_context
            )
            
            response = await self.model.ainvoke(formatted_prompt)
            parsed_output = self.recovery_parser.parse(response.content)
            return parsed_output.model_dump()
            
        except Exception as e:
            logger.error(f"Recovery plan generation failed: {e}", exc_info=True)
            return {"tool": "finish", "input": {"reason": f"Recovery failed: {error}"}}


class StepParameters(BaseModel):
    """Model for step parameters."""
    url: Optional[str] = None
    selector: Optional[str] = None
    text: Optional[str] = None

    @validator('url')
    def validate_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class Step(BaseModel):
    """Model for a single step in the plan."""
    action: str = Field(..., description="The action to perform (navigate, click, or type)")
    parameters: StepParameters = Field(..., description="Parameters for the action")
    description: str = Field(..., description="Description of what this step does")

    @validator('action')
    def validate_action(cls, v):
        valid_actions = {'navigate', 'click', 'type'}
        if v not in valid_actions:
            raise ValueError(f"Action must be one of {valid_actions}")
        return v

    @validator('parameters')
    def validate_parameters(cls, v, values):
        action = values.get('action')
        if action == 'navigate' and not v.url:
            raise ValueError("URL is required for navigate action")
        elif action == 'click' and not v.selector:
            raise ValueError("Selector is required for click action")
        elif action == 'type' and (not v.selector or not v.text):
            raise ValueError("Selector and text are required for type action")
        return v


class Plan(BaseModel):
    """Model for the complete plan."""
    steps: List[Step] = Field(..., description="List of steps to execute")
    reasoning: str = Field(..., description="Explanation of the plan")


class PlanResponse(BaseModel):
    """Model for the complete plan response."""
    status: str = Field(..., description="Status of the plan generation")
    plan: Plan = Field(..., description="The generated plan")
    error: Optional[str] = Field(None, description="Error message if status is not success")

    @validator('status')
    def validate_status(cls, v):
        if v not in {'success', 'error'}:
            raise ValueError("Status must be either 'success' or 'error'")
        return v


def format_url(url: str) -> str:
    """Format URL to ensure consistency."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse(f"https://{url}")
        return urlunparse(parsed)
    except Exception as e:
        logger.error(f"Error formatting URL {url}: {str(e)}")
        return url


def extract_json_from_response(response_text: str, max_retries: int = 3) -> Optional[Dict]:
    """Extract and validate JSON from LLM response with retries."""
    for attempt in range(max_retries):
        try:
            # Try to find JSON in the response
            json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in response (attempt {attempt + 1}/{max_retries})")
                continue

            json_str = json_match.group(1)
            # Try to parse the JSON
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in response (attempt {attempt + 1}/{max_retries}): {str(e)}")
                # Try to fix common JSON issues
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error(f"Error extracting JSON (attempt {attempt + 1}/{max_retries}): {str(e)}")
            continue

    return None


def validate_plan_data(plan_data: Dict) -> Optional[PlanResponse]:
    """Validate and convert plan data to PlanResponse model."""
    try:
        return PlanResponse(**plan_data)
    except Exception as e:
        logger.error(f"Error validating plan data: {str(e)}")
        return None


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    # Primary provider config
    primary_provider: str = os.environ.get("LLM_PROVIDER", "nvidia") # Default to nvidia
    primary_model: str = os.environ.get("MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1")
    primary_base_url: Optional[str] = os.environ.get("NVIDIA_API_BASE_URL")  # No default, let code handle it
    primary_api_key: Optional[str] = os.environ.get("NVIDIA_API_KEY")

    # Fallback providers (Example: OpenAI)
    fallback_providers: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "openai": {
            "model_name": os.environ.get("OPENAI_MODEL_NAME", "gpt-4-turbo-preview"),
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_BASE_URL")
        }
    })

    # Common parameters - ensure these are sourced from env vars if possible
    temperature: float = float(os.environ.get("MODEL_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.environ.get("MODEL_MAX_TOKENS", "4096"))
    streaming: bool = os.environ.get("MODEL_ENABLE_STREAMING", "True").lower() == "true"
    timeout: int = int(os.environ.get("MODEL_TIMEOUT", "15"))
    top_p: float = float(os.environ.get("MODEL_TOP_P", "0.9"))
    top_k: int = int(os.environ.get("MODEL_TOP_K", "50"))
    repetition_penalty: float = float(os.environ.get("MODEL_REPETITION_PENALTY", "1.1"))
    max_retries: int = int(os.environ.get("MODEL_MAX_RETRIES", "3"))
    retry_delay: float = float(os.environ.get("MODEL_RETRY_DELAY", "1.0"))
    batch_size: int = int(os.environ.get("MODEL_BATCH_SIZE", "4"))

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Validate primary provider
        if not self.primary_provider:
            raise ValueError("LLM provider must be specified")
            
        # Validate model name
        if not self.primary_model:
            raise ValueError("Model name must be specified")
            
        # Validate API key for NVIDIA
        if self.primary_provider.lower() == "nvidia":
            if not self.primary_api_key:
                raise ValueError("NVIDIA_API_KEY must be set for NVIDIA provider")
                
        # Normalize temperature
        self.temperature = max(0.0, min(1.0, self.temperature))
        
        # Ensure reasonable token limits
        self.max_tokens = max(1, min(8192, self.max_tokens))
        
        # Ensure reasonable timeout
        self.timeout = max(1, self.timeout)
        
        # Normalize top_p
        self.top_p = max(0.0, min(1.0, self.top_p))
        
        # Ensure reasonable top_k
        self.top_k = max(1, min(100, self.top_k))
        
        # Ensure reasonable repetition penalty
        self.repetition_penalty = max(1.0, min(2.0, self.repetition_penalty))
        
        # Ensure reasonable retry settings
        self.max_retries = max(0, self.max_retries)
        self.retry_delay = max(0.1, self.retry_delay)
        
        # Ensure reasonable batch size
        self.batch_size = max(1, min(32, self.batch_size))


class LLM:
    """Unified Language Model interface."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM based on configuration."""
        self.config = config
        self._base_llm = self._init_llm()
        self._adapter = LangChainAdapter(self._base_llm)
    
    def _init_llm(self) -> BaseChatModel:
        """Initialize and return a LangChain chat model instance."""
        provider = self.config.primary_provider.lower()
        model_name = self.config.primary_model
        api_key = os.environ.get("GOOGLE_API_KEY") if provider == "google" else os.environ.get("NVIDIA_API_KEY")
        base_url = self.config.primary_base_url

        model_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }
        
        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    **model_kwargs
                )
            except ImportError:
                raise ImportError("Please install langchain-google-genai to use Google's models")
                
        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=api_key,
                    openai_api_base=base_url,
                    **model_kwargs
                )
            except ImportError:
                raise ImportError("Please install langchain-openai to use OpenAI's models")
                
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        return llm
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan using the adapter."""
        return await self._adapter.generate_plan(task=task, context=context)
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response using the adapter."""
        return await self._adapter.generate_response(task=task, plan=plan, context=context)
    
    async def interpret_command(self, command: str) -> Dict[str, Any]:
        """Interpret a command using the adapter."""
        return await self._adapter.interpret_command(command=command)
    
    async def generate_recovery_plan(self, error: str, action: Dict[str, Any], dom_context: str) -> Dict[str, Any]:
        """Generate a recovery plan using the adapter."""
        return await self._adapter.generate_recovery_plan(error=error, action=action, dom_context=dom_context)

    def get_langchain_llm(self) -> BaseChatModel:
        """Returns the base LangChain LLM instance."""
        # Return the base model, not the structured one, for general use if needed outside
        return self._base_llm 