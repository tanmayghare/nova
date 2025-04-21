from __future__ import annotations

"""Language model integration."""

import os
import logging
import json
import re
import base64
import httpx

from typing import Any, Dict, List, Optional, Protocol, cast
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator, field_validator

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

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
    tool: str = Field(..., description="The name of the tool to use. Must be one of the available browser or web search tools.")
    input: Dict[str, Any] = Field(default_factory=dict, description="The input parameters for the tool.")

    @field_validator('tool')
    def tool_must_be_valid(cls, v):
        valid_tools = {
            'navigate', 'click', 'type', 'get_text', 'wait', 
            'screenshot', 'get_dom_snapshot', 
            'web_search', 
            'finish'
        }
        if v not in valid_tools:
            raise ValueError(f"Invalid tool name: {v}. Must be one of {valid_tools}")
        return v

    @field_validator('input')
    def validate_input_parameters(cls, v, info):
        tool = info.data.get('tool')
        if tool == 'navigate':
            if 'url' not in v:
                raise ValueError("URL is required for navigate tool")
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
    
    async def generate_plan(self, task: str, context: str, history: List[str], dom_content: Optional[str], available_tools: List[Dict]) -> tuple[str, float, List[Dict[str, Any]]]:
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
    
    def __init__(self, model: BaseChatModel, config: LLMConfig):
        """Initialize with a LangChain model and configuration."""
        self.model = model
        self.config = config
        # Initialize output parsers
        self.plan_parser = PydanticOutputParser(pydantic_object=PlanGenerationOutput)
        self.action_parser = PydanticOutputParser(pydantic_object=ActionOutput)
        self.recovery_parser = PydanticOutputParser(pydantic_object=RecoveryPlanOutput)
        self.command_parser = PydanticOutputParser(pydantic_object=CommandInterpretationOutput)
    
    async def generate_plan(self, task: str, context: str, history: List[str], dom_content: Optional[str], available_tools: List[Dict], screenshot_bytes: Optional[bytes] = None) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan for executing a task using structured output, history, DOM, and potentially vision."""
        
        # --- Hardcoded Few-Shot Examples --- 
        # Escape curly braces in example JSON inputs using double braces {{ }}
        # This prevents Python's string formatter from misinterpreting them.

        few_shot_examples = """
        Here are some examples of how to structure your plan:

        Example 1: Clicking a Vague Icon Button (Vision Focus)
        Input State Description: "History shows we just navigated to 'example-dashboard.com'. The DOM is complex, but the screenshot clearly shows a user profile icon in the top-right corner which doesn't have a clear text label or unique ID in the DOM. Task is 'Log out'."
        Desired Output JSON: {{ "tool": "click", "input": {{ "selector": "img[alt*='profile'], .user-icon, #profile-button" }} }}

        Example 2: Extracting Text from Standard Location (DOM Focus)
        Input State Description: "History shows we searched DuckDuckGo for 'MVP development'. The DOM clearly shows typical search result structure with `h2` tags containing result titles inside `article` elements. Task is 'Get the title of the first result'."
        Desired Output JSON: {{ "tool": "get_text", "input": {{ "selector": "article h2 a" }} }}
        
        Example 3: Finishing the Task
        Input State Description: "History shows we navigated to example.com and then successfully used get_text(selector='h1') which returned 'Example Domain'. Task is 'Navigate to example.com, find the main heading, and tell me what it is.'"
        Desired Output JSON: {{ "tool": "finish", "input": {{ "output": "Example Domain" }} }}
        """
        # --- End Few-Shot Examples ---
        
        try:
            # --- Prepare common prompt elements --- 
            history_str = "\n".join(history) if history else "No history yet."
            tools_str = json.dumps(available_tools, indent=2)
            dom_str = dom_content if dom_content else "No DOM content available."
            # Truncate potentially very long DOM string for the prompt
            if len(dom_str) > 3000: # Limit DOM context in prompt
                dom_str = dom_str[:3000] + "... (truncated)"
            
            # Define the explicit output format instructions manually
            # This avoids relying on PydanticOutputParser.get_format_instructions() which caused KeyErrors
            # Escape curly braces {{ }} within the example JSON to prevent formatting errors
            output_format_instructions = """
            **Output Format:**
            You MUST respond with a JSON object containing the following fields:
            - "thought_process": "Your reasoning for choosing the next action."
            - "plan": [ {{ "tool": "<tool_name>", "input": {{ <tool_parameters> }} }} ] (A list containing EXACTLY ONE tool call object)
            - "confidence": <float between 0.0 and 1.0>

            Example JSON Output:
            ```json
            {{
                "thought_process": "The task is to find the main heading. The DOM shows an H1 element which is likely it. I will use get_text.",
                "plan": [ 
                    {{ 
                        "tool": "get_text", 
                        "input": {{ "selector": "h1" }} 
                    }}
                ],
                "confidence": 0.9
            }}
            ```
            """
            
            system_prompt_template = f"""You are a task execution agent that generates a single next step for completing a task.
            **IMMEDIATE GOAL:** Your FIRST step MUST be directly aimed at fulfilling the user's Task description provided below. If the Task description is clear (e.g., 'Navigate to X', 'Search for Y'), perform that action directly.
            **PRIMARY OBJECTIVE:** You MUST focus on completing the original user Task description. Use History, DOM, Context, and Search Results ONLY as supporting information to achieve the original Task. Do not get sidetracked.
            Your goal is to complete the user's task by interacting with a web browser or searching the web.
            Analyze the available inputs (task description, history, DOM, screenshot if available) carefully.
            
            Available Tools:
            ```json
            {{available_tools}}
            ```

            Few-Shot Examples:
            {few_shot_examples}
            
            CRITICAL RULES:
            1. Generate EXACTLY ONE step/tool call in the plan.
            2. If the Task description gives a clear starting action (like 'Navigate' or 'Search'), perform that first. Do NOT web search about how to perform the task itself.
            3. Choose the most appropriate tool based on the task, history, DOM, and vision (if available).
            4. Provide necessary parameters for the chosen tool.
            5. **CRITICAL:** If the result of the PREVIOUS action provides the complete answer to the original user task, you MUST use the 'finish' tool NOW with the answer in the 'output' argument. DO NOT take any other action.
            6. Otherwise, if the task is not yet complete, choose a browser or web_search tool. Use `web_search` ONLY when external information is explicitly needed and cannot be found via browser interaction.
            7. Analyze the DOM content and history carefully before deciding.
            8. Avoid redundant actions. If you just navigated to a page, don't navigate again unless necessary. If you just got the DOM, don't get it again unless an action modified the page.
            9. Confidence score MUST be between 0.0 and 1.0.
            
            {output_format_instructions} 
            
            Current Conversation History (Oldest first):
            {{history}}
            
            Current DOM Content (potentially truncated):
            ```html
            {{dom_content}}
            ```
            
            Previous Context/Memory:
            {{context}}
            """
            
            human_prompt_template = "Task: {{task}}\nBased on the history, DOM, screenshot (if available) and available tools, what is the single next step?"
            
            # --- Service Type Specific Logic --- 
            if self.config.service_type == "api":
                logger.info("Generating plan using NIM API (Llama 4 Scout)...")
                api_key = os.environ.get("NVIDIA_API_KEY")
                if not api_key:
                    raise ValueError("NVIDIA_API_KEY environment variable not set for API call.")
                
                headers = {
                  "Authorization": f"Bearer {api_key}",
                  "Accept": "application/json"
                }

                # Prepare prompt text for API, ensuring the manually defined instructions are included
                system_prompt_formatted = system_prompt_template.format(
                    available_tools=tools_str, 
                    history=history_str, 
                    dom_content=dom_str, 
                    context=context,
                    output_format_instructions=output_format_instructions # Pass manual instructions
                )
                human_prompt_formatted = human_prompt_template.format(task=task)
                prompt_text_for_api = f"{system_prompt_formatted}\n\n{human_prompt_formatted}"
                
                # Construct message payload, including image if available
                messages = []
                user_content_parts = [prompt_text_for_api]
                if screenshot_bytes:
                    try:
                        image_b64 = base64.b64encode(screenshot_bytes).decode()
                        # Simple check for size, API might have different limits
                        if len(image_b64) < 180_000: 
                             user_content_parts.append(f' <img src="data:image/png;base64,{image_b64}" />')
                             logger.info("Adding screenshot to API request.")
                        else:
                             logger.warning("Screenshot too large to embed directly in API request, skipping.")
                    except Exception as e:
                        logger.error(f"Error encoding screenshot: {e}", exc_info=True)
                
                messages.append({"role": "user", "content": "".join(user_content_parts)})

                payload = {
                  "model": self.config.api_model,
                  "messages": messages,
                  "max_tokens": self.config.max_tokens,
                  "temperature": self.config.temperature,
                  "top_p": self.config.top_p,
                  "stream": False
                }
                
                async with httpx.AsyncClient() as client:
                    logger.debug(f"Sending request to NIM API: {self.config.api_endpoint}")
                    response = await client.post(
                        self.config.api_endpoint,
                        headers=headers, 
                        json=payload, 
                        timeout=self.config.timeout + 5 # Add buffer to client timeout
                    )
                
                response.raise_for_status() # Raise exception for non-2xx status codes
                response_json = response.json()
                logger.debug(f"Received response from NIM API: {response_json}")
                
                # Extract the plan text from the API response
                plan_text = response_json['choices'][0]['message']['content']
                
            elif self.config.service_type == "container":
                logger.info("Generating plan using container LLM...")
                if not self.model:
                     raise ValueError("Container LLM (self.model) not initialized.")
                     
                # Create prompt template for LangChain model 
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_template), 
                    ("human", human_prompt_template)
                ])
                
                # Format the prompt with the necessary context, including manual format instructions
                formatted_prompt = prompt.format_messages(
                    available_tools=tools_str,
                    history=history_str,
                    dom_content=dom_str,
                    context=context,
                    task=task,
                    output_format_instructions=output_format_instructions # Pass manual instructions
                )
                
                # Get the model's response via LangChain
                lc_response = await self.model.ainvoke(formatted_prompt)
                plan_text = cast(str, lc_response.content) # Cast to string
            
            else:
                # Should be caught earlier, but defensive check
                raise ValueError(f"Unsupported service type for plan generation: {self.config.service_type}")

            # Parse the response text (common to both API and container)
            logger.debug(f"Raw plan text to parse: {plan_text}")
            parsed_output = self.plan_parser.parse(plan_text)
            
            # Extract data from the validated Pydantic object
            thought = parsed_output.thought_process
            confidence = parsed_output.confidence
            plan_steps_dict = [step.model_dump() for step in parsed_output.plan]
            
            # Log success
            logger.info(f"Successfully generated structured plan: confidence={confidence:.2f}, steps={len(plan_steps_dict)}")
            logger.debug(f"Generated thought: {thought[:200]}...")
            logger.debug(f"Generated plan steps: {plan_steps_dict}")

            return thought, confidence, plan_steps_dict

        except httpx.RequestError as e:
             logger.error(f"NIM API request failed: {e}", exc_info=True)
             return "API Request Error", 0.0, []
        except httpx.HTTPStatusError as e:
             logger.error(f"NIM API returned error status {e.response.status_code}: {e.response.text}", exc_info=True)
             return f"API Status Error {e.response.status_code}", 0.0, []
        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            return "Plan Generation Error", 0.0, []
    
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
    # Primary provider config (Keep for potential future use with other container providers)
    primary_provider: str = os.environ.get("LLM_PROVIDER", "nvidia") 
    
    # Service Type Selection
    service_type: str = os.environ.get("LLM_SERVICE_TYPE", "container").lower() # Default to container

    # Container specific (Used if service_type == 'container')
    container_model: str = os.environ.get("MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1")
    container_base_url: Optional[str] = os.environ.get("NVIDIA_API_BASE_URL")
    
    # API specific (Used if service_type == 'api')
    api_endpoint: str = os.environ.get("LLM_API_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")
    api_model: str = os.environ.get("LLM_API_MODEL", "meta/llama-4-scout-17b-16e-instruct")

    # API Key (Potentially used by both, ensure it's set)
    api_key: Optional[str] = os.environ.get("NVIDIA_API_KEY") 

    # Fallback providers (Keep for potential future use)
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
        # Validate service type
        if self.service_type not in ["container", "api"]:
             raise ValueError(f"Invalid LLM_SERVICE_TYPE: '{self.service_type}'. Must be 'container' or 'api'.")

        # Validate provider (relevant mainly for container type if using non-NVIDIA containers later)
        if not self.primary_provider:
            raise ValueError("LLM provider must be specified")
            
        # Validate settings based on service type
        if self.service_type == 'container':
            if not self.container_model:
                raise ValueError("MODEL_NAME must be specified for container service type")
            # Base URL might be optional depending on container setup, but key is usually needed
            if self.primary_provider.lower() == "nvidia" and not self.api_key:
                 logger.warning("NVIDIA_API_KEY is typically required for NVIDIA NIM containers. Ensure it's set if needed.")
                 # raise ValueError("NVIDIA_API_KEY must be set for NVIDIA provider")
        elif self.service_type == 'api':
             if not self.api_endpoint:
                  raise ValueError("LLM_API_ENDPOINT must be specified for api service type")
             if not self.api_model:
                  raise ValueError("LLM_API_MODEL must be specified for api service type")
             if not self.api_key:
                  raise ValueError("NVIDIA_API_KEY (or relevant API key) must be set for api service type")
            
        # Validate API key (General check, specific needs handled above)
        # if not self.api_key:
        #     raise ValueError("NVIDIA_API_KEY (or equivalent) must be set")
                
        # Normalize common parameters
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
        """Initialize LLM based on configuration service type."""
        self.config = config
        self._base_llm: Optional[BaseChatModel] = None # Initialize as None
        
        if self.config.service_type == "container":
            logger.info(f"Initializing LLM for container service: Provider='{self.config.primary_provider}', Model='{self.config.container_model}'")
            self._base_llm = self._init_container_llm() # Use specific init method
            if self._base_llm:
                 self._adapter = LangChainAdapter(self._base_llm, self.config) # Pass config to adapter
            else:
                 # Handle case where container LLM failed to initialize
                 logger.error("Failed to initialize container LLM. Adapter not created.")
                 self._adapter = None # Or raise error?
        elif self.config.service_type == "api":
            logger.info(f"Initializing LLM for API service: Endpoint='{self.config.api_endpoint}', Model='{self.config.api_model}'")
            # For API type, the LangChain model might not be used directly.
            # The adapter will handle API calls. We still need the adapter instance.
            # Pass a dummy or None model, but ensure adapter gets the config.
            self._adapter = LangChainAdapter(model=None, config=self.config) 
        else:
            # This case should be caught by __post_init__, but as a safeguard:
             raise ValueError(f"Unsupported LLM service type: {self.config.service_type}")

        if not self._adapter:
             raise RuntimeError("LLM Adapter could not be initialized.")

    def _init_container_llm(self) -> Optional[BaseChatModel]: # Renamed and added return type
        """Initialize and return a LangChain chat model instance for container service."""
        provider = self.config.primary_provider.lower()
        model_name = self.config.container_model
        api_key = self.config.api_key # Use the unified api_key field
        base_url = self.config.container_base_url

        model_kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            # Add other relevant kwargs if ChatNVIDIA supports them
        }
        
        llm: Optional[BaseChatModel] = None
        
        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Ensure correct API key is used if Google becomes primary
                google_api_key = os.environ.get("GOOGLE_API_KEY") 
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=google_api_key,
                    **model_kwargs
                )
            except ImportError:
                raise ImportError("Please install langchain-google-genai to use Google's models")
                
        elif provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
                # Ensure correct API key/base_url are used if OpenAI becomes primary
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                openai_base_url = os.environ.get("OPENAI_BASE_URL")
                llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=openai_api_key,
                    base_url=openai_base_url,
                    **model_kwargs
                )
            except ImportError:
                raise ImportError("Please install langchain-openai to use OpenAI's models")
        
        elif provider == "nvidia":
            try:
                from langchain_nvidia_ai_endpoints import ChatNVIDIA
                # Use the configuration directly
                llm = ChatNVIDIA(
                     model=model_name, 
                     nvidia_api_key=api_key,
                     base_url=base_url, # Pass the NIM container base URL
                     **model_kwargs
                 )
                logger.info(f"Initialized ChatNVIDIA with model='{model_name}', base_url='{base_url}'")
            except ImportError:
                 raise ImportError("Please install langchain-nvidia-ai-endpoints to use NVIDIA NIMs.")
            except Exception as e:
                 logger.error(f"Failed to initialize ChatNVIDIA: {e}", exc_info=True)
                 # Decide if we should raise or return None
                 # raise # Re-raise for now to make failure explicit
                 return None # Or return None if initialization failure should be handled gracefully
                
        else:
            # This error should now only trigger for truly unsupported providers
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        return llm
    
    async def generate_plan(self, task: str, context: str, history: List[str], dom_content: Optional[str], available_tools: List[Dict], screenshot_bytes: Optional[bytes] = None) -> tuple[str, float, List[Dict[str, Any]]]: # Added screenshot_bytes
        """Generate a plan using the adapter."""
        if not self._adapter:
            raise RuntimeError("LLM Adapter not initialized.")
        return await self._adapter.generate_plan(task=task, context=context, history=history, dom_content=dom_content, available_tools=available_tools, screenshot_bytes=screenshot_bytes)
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response using the adapter."""
        if not self._adapter:
            raise RuntimeError("LLM Adapter not initialized.")
        return await self._adapter.generate_response(task=task, plan=plan, context=context)
    
    async def interpret_command(self, command: str) -> Dict[str, Any]:
        """Interpret a command using the adapter."""
        if not self._adapter:
            raise RuntimeError("LLM Adapter not initialized.")
        return await self._adapter.interpret_command(command=command)
    
    async def generate_recovery_plan(self, error: str, action: Dict[str, Any], dom_context: str) -> Dict[str, Any]:
        """Generate a recovery plan using the adapter."""
        if not self._adapter:
            raise RuntimeError("LLM Adapter not initialized.")
        return await self._adapter.generate_recovery_plan(error=error, action=action, dom_context=dom_context)

    def get_langchain_llm(self) -> BaseChatModel:
        """Returns the base LangChain LLM instance."""
        # Return the base model, not the structured one, for general use if needed outside
        return self._base_llm 