from __future__ import annotations

"""Language model integration."""

import json
import logging
import re
import asyncio
import os

from typing import Any, Dict, List, Optional, Protocol, Union, AsyncIterator
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from pydantic import BaseModel, Field, validator

from .providers.llama import LlamaModel

logger = logging.getLogger(__name__)


class LanguageModel(Protocol):
    """Protocol for language models."""
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, List[Dict[str, Any]]]:
        """Generate a plan (including thought) for executing a task."""
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
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan for executing a task, including the thought process and confidence score.
        
        Args:
            task: The task description
            context: Additional context for the task
            
        Returns:
            A tuple containing the thought (str), confidence (float), and the list of steps (List[Dict]).
        """
        prompt = f"""You are Nova, an intelligent browser automation agent designed to accomplish tasks by interacting with web pages using available tools.

Your goal is to generate a plan to accomplish the given task based on the current context, **including the structure of the current web page**. Follow the Thought-Action format, **including a confidence score for your proposed action**:

1.  **Thought:** Briefly explain your reasoning process. 
    - Analyze the `Initial Task Context` and `Recent Execution History`.
    - **Critically examine the `Current Page Structure (After Previous Action)` JSON.** 
    - Note the outcome (`observation`) of the last step in the history. 
        - **If the last observation status was 'error':** Analyze the `error` message. Based on the error and the current page structure, decide on a corrective action (e.g., try a different selector, wait, try a different tool) or conclude with `finish` if recovery is impossible.
        - **If the last observation status was 'low_confidence_retry':** Re-evaluate the goal using the potentially added 'Extended Context (Full HTML)' and propose a higher-confidence action or `finish`.
        - **Otherwise (success or first step):** Determine the single best next step towards the `Task` goal based on the current page structure and history.
    - If the overall task goal is achieved according to the history and current page state, explain why and use the 'finish' tool.
2.  **Confidence Score:** Provide a numerical score (0.0 to 1.0) indicating your confidence that the proposed action is correct and will succeed towards the goal. 
    - Base confidence on: clarity of the goal, uniqueness/reliability of selectors (if applicable), consistency with history, likelihood of achieving the task objective with this step.
    - Use lower scores if selectors are ambiguous, the action seems risky, or the goal is unclear.
3.  **Action:** Generate the plan as a JSON array containing the **single next step** (a tool call or the 'finish' action).

Task: {task}

Context:
```
{context}
```
*Note: The context above includes initial context, the current page structure captured after the previous action completed, and recent history. Each history entry includes 'thought', 'action', 'confidence' (if available), and 'observation'. The observation contains the action's 'status' ('success', 'error', 'low_confidence_retry', 'halted'), 'result' or 'error', and potentially a 'screenshot' file path. If the previous step had low confidence, the context might also include an 'Extended Context (Full HTML from Previous Step)' section.* 

Available tools:
- navigate: Navigate to a URL (input: {{'url': '...'}})
- click: Click an element matching a selector (input: {{'selector': '...'}})
- type: Type text into an element (input: {{'selector': '...', 'text': '...'}})
- wait: Wait for an element to appear (input: {{'selector': '...', 'timeout': ...}})
- screenshot: Take a screenshot (input: {{'path': '...'}})
- finish: Use this tool **only** when the task goal is fully achieved. (input: {{'reason': '...'}} - Optional reason for completion)
# Add other tools dynamically if needed

Output Format:
Your final output **must** be a valid JSON object containing three keys: 'thought' (string), 'confidence' (float between 0.0 and 1.0), and 'plan' (JSON array with one step).

Example (Action Step):
```json
{{
  "thought": "The task is to click the login button. Looking at the `Current Page Structure (After Previous Action)`, I see a button with `text: 'Login'` and a unique `id: 'login-btn'`. This seems unambiguous.",
  "confidence": 0.95,
  "plan": [
    {{\"tool\": \"click\", \"input\": {{\"selector\": \"#login-btn\"}}}}
  ]
}}
```

Example (Corrective Step after Error):
```json
{{
  "thought": "The previous action `click` with selector '#submit' failed with error 'Timeout waiting for element'. The current page structure still shows the button exists. I will try waiting for the element to be clickable first, then click again.",
  "confidence": 0.8,
  "plan": [
    {{\"tool\": \"wait\", \"input\": {{"selector": \"#submit\", \"timeout\": 15}}}}
    // Note: Ideally the LLM would then plan the click in the *next* step after the wait succeeds.
    // For now, we expect one action per step.
  ]
}}
```

Example (Finish Step):
```json
{{
  "thought": "The user asked for the page title, and the context shows the title is 'Example Domain'. The task is complete.",
  "confidence": 1.0,
  "plan": [
    {{\"tool\": \"finish\", \"input\": {{\"reason\": \"Found the page title as requested.\"}}}}
  ]
}}
```

Generate the thought, confidence score, and the single next step (or finish action) for the task:"""

        try:
            response = await self.model.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON object from response
            plan_json = None
            try:
                # Priority 1: Look for ```json ... ``` block
                json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if json_block_match:
                    plan_json = json_block_match.group(1).strip()
                    logger.debug("Extracted JSON using ```json block.")
                else:
                    # Priority 2: Look for the first generic { ... } block
                    generic_match = re.search(r"\{\s*.*?\s*\}", response_text, re.DOTALL)
                    if generic_match:
                        plan_json = generic_match.group(0).strip()
                        logger.debug("Extracted JSON using generic {.*?} block.")
                    else:
                         raise ValueError("No JSON object found in response text using ```json or {.*?} patterns.")

                logger.debug(f"Extracted JSON string: {plan_json[:500]}...")
                parsed_data = json.loads(plan_json)
                
                if not isinstance(parsed_data, dict):
                    raise ValueError("Parsed JSON is not a dictionary")

                # Extract thought, confidence, and plan
                thought = parsed_data.get("thought", "") 
                plan_steps = parsed_data.get("plan", []) 
                confidence = 0.0 # Default confidence
                raw_confidence = parsed_data.get("confidence")
                if isinstance(raw_confidence, (float, int)):
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                elif isinstance(raw_confidence, str):
                    try:
                        confidence = max(0.0, min(1.0, float(raw_confidence)))
                    except ValueError:
                        logger.warning(f"Could not parse confidence score string: '{raw_confidence}'. Defaulting to 0.0.")
                elif raw_confidence is not None:
                     logger.warning(f"Unexpected type for confidence score: {type(raw_confidence)}. Defaulting to 0.0.")

                if not isinstance(plan_steps, list):
                    raise ValueError("The 'plan' field must contain a list of steps")
                
                # Normalize and validate steps (no major change needed here for finish tool)
                normalized_steps = []
                for step in plan_steps:
                    if not isinstance(step, dict):
                        logger.warning(f"Skipping invalid step format: {step}")
                        continue
                        
                    tool_name = step.get("tool")
                    input_params = step.get("input", {})

                    if not tool_name:
                         logger.warning(f"Skipping step missing 'tool' key: {step}")
                         continue 
                    
                    if not isinstance(input_params, dict):
                         # Allow non-dict input for finish if needed, but generally expect dict
                         if tool_name != "finish":
                             logger.warning(f"Input for tool '{tool_name}' is not a dictionary: {input_params}. Using empty input.")
                             input_params = {} 
                         elif input_params is None: # Allow null input for finish
                              input_params = {}
                         elif not isinstance(input_params, dict): # Treat other non-dicts as error for finish? Or empty?
                              logger.warning(f"Non-dict input for finish tool: {input_params}. Using empty input.")
                              input_params = {}

                    normalized_step = {
                        "tool": tool_name,
                        "input": input_params
                    }
                    normalized_steps.append(normalized_step)
                
                logger.info(f"Successfully parsed thought, confidence={confidence:.2f}, and {len(normalized_steps)} plan steps.")
                # Return the 3-tuple
                return thought, confidence, normalized_steps 
                
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Failed to parse LLM response JSON: {e}", exc_info=True)
                if plan_json:
                     logger.error(f"JSON string that failed parsing: {plan_json}")
                else:
                     logger.error(f"Raw response potentially causing error: {response_text}")
                # Return empty thought, 0.0 confidence, empty plan on failure
                return "", 0.0, [] 

        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            # Return empty on LLM communication failure
            return "", 0.0, [] 
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results (ReAct history).
        
        Args:
            task: The original task description
            plan: The full ReAct action history (list of thought/action/observation dicts)
            context: The final context string used before generating this response (optional context)
            
        Returns:
            Generated response summarizing the execution
        """
        # Use the action_history (passed as 'plan') as the primary source
        history_json = json.dumps(plan, indent=2) 
        
        prompt = f"""You are Nova, an intelligent browser automation agent. The following task was attempted:

Task: {task}

The execution involved the following sequence of thoughts, actions, and observations (ReAct History):
```json
{history_json}
```

Generate a concise natural language response for the user that:
1. Summarizes the key actions taken based on the history.
2. Mentions any significant errors encountered during execution (from observations).
3. States the final outcome or result (if discernible from the history or cumulative results).

Response:"""

        try:
            response = await self.model.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            raise # Re-raise for now, agent might handle it


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
    primary_provider: str = os.environ.get("LLM_PROVIDER", "nvidia_nim") # Default to nim
    primary_model: str = os.environ.get("MODEL_NAME", "nvidia/llama-3.3-nemotron-super-49b-v1")
    primary_base_url: Optional[str] = os.environ.get("NIM_API_BASE_URL")  # No default, let code handle it
    primary_api_key: Optional[str] = os.environ.get("NVIDIA_NIM_API_KEY")

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
    timeout: int = int(os.environ.get("MODEL_TIMEOUT", "30"))
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
            
        # Validate API key for NIM
        if self.primary_provider.lower() in ["nim", "nvidia_nim"]:
            if not self.primary_api_key:
                raise ValueError("NVIDIA_NIM_API_KEY must be set for NIM provider")
                
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
    """Abstraction layer for different LLM providers."""

    def __init__(self, config: LLMConfig):
        """Initialize the LLM wrapper, selecting the provider."""
        self.config = config
        self.llm_instance = self._init_llm() # Store the specific provider instance
        self._parser = JsonOutputParser() # Added parser initialization back

    def _init_llm(self) -> BaseChatModel:
        """Initialize the appropriate Langchain LLM provider based on config."""
        provider = self.config.primary_provider.lower()
        logger.info(f"Initializing LLM with provider: {provider}")

        # *** Use LangChain ChatNVIDIA for NIM provider ***
        if provider == "nvidia_nim" or provider == "nim":
            try:
                logger.info("Attempting Langchain ChatNVIDIA initialization...")
                from langchain_nvidia_ai_endpoints import ChatNVIDIA
                
                # Ensure API key is fetched correctly
                api_key = self.config.primary_api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
                if not api_key:
                    raise ValueError("NVIDIA API Key not found in environment variables or config.")
                
                # Ensure base_url is correctly formatted with /v1
                base_url = self.config.primary_base_url
                if not base_url:
                    base_url = "http://localhost:8000"
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url.rstrip('/')}/v1"
                logger.info(f"Using NIM Base URL: {base_url}")
                
                # Ensure model name is correctly formatted
                model_name = self.config.primary_model
                if not model_name.startswith("nvidia/"):
                    model_name = f"nvidia/{model_name}"
                logger.info(f"Using NIM Model: {model_name}")
                
                return ChatNVIDIA(
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    streaming=self.config.streaming,
                    timeout=self.config.timeout,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    max_retries=self.config.max_retries,
                    retry_delay=self.config.retry_delay,
                    batch_size=self.config.batch_size
                )
            except ImportError:
                logger.error("Langchain NVIDIA AI Endpoints library not installed. Cannot initialize ChatNVIDIA. Please run `pip install langchain-nvidia-ai-endpoints`.")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize ChatNVIDIA: {e}", exc_info=True)
                raise ValueError(f"Failed to initialize primary LLM provider '{provider}': {str(e)}")

        # --- Fallback to other configured providers ---
        logger.info(f"Primary provider '{provider}' not specified as NIM. Attempting fallback providers...")
        for fallback_provider_name, fallback_config in (self.config.fallback_providers or {}).items():
            try:
                logger.info(f"Attempting fallback provider: {fallback_provider_name}")
                if fallback_provider_name.lower() == "openai":
                    from langchain_openai import ChatOpenAI
                    return ChatOpenAI(
                        model_name=fallback_config["model_name"],
                        api_key=fallback_config["api_key"],
                        base_url=fallback_config["base_url"],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        streaming=self.config.streaming,
                        timeout=self.config.timeout
                    )
                elif fallback_provider_name.lower() == "google":
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    return ChatGoogleGenerativeAI(
                        model=fallback_config["model_name"],
                        google_api_key=fallback_config["api_key"],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        streaming=self.config.streaming,
                        timeout=self.config.timeout
                    )
                # Add other langchain providers if needed
            except ImportError:
                 logger.warning(f"Required library for Langchain provider '{fallback_provider_name}' not installed. Skipping.")
                 continue
            except Exception as e:
                logger.warning(f"Failed to initialize Langchain {fallback_provider_name} LLM: {e}. Skipping.")
                continue

        logger.error("No valid LLM provider could be initialized based on configuration.")
        raise ValueError("No valid LLM provider could be initialized")


    # --- Methods using self.llm_instance --- 
    # These methods now assume self.llm_instance is always a BaseChatModel

    async def generate(
        self,
        prompt: Union[str, ChatPromptTemplate],
        input_variables: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
         """Generate text using the selected LangChain LLM."""
         if isinstance(prompt, str):
             message = HumanMessage(content=prompt)
         else: # Assume ChatPromptTemplate
             message = prompt.format_messages(**(input_variables or {}))

         if stream:
             # Check if the model supports streaming (most ChatModels do)
             if hasattr(self.llm_instance, 'astream'):
                 return self.llm_instance.astream(message)
             else:
                 logger.warning("Streaming requested but LLM instance does not support astream. Falling back to non-streaming.")
                 response = await self.llm_instance.ainvoke(message)
                 async def async_generator(): # Wrap single result in async generator
                     yield response.content
                 return async_generator()
         else:
             response = await self.llm_instance.ainvoke(message)
             return response.content

    async def generate_plan(
        self,
        task: str,
        context: str,
    ) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan using the LangChain LLM via LangChainAdapter."""
        # Use the LangChainAdapter to handle the specific prompting and parsing
        adapter = LangChainAdapter(self.llm_instance)
        return await adapter.generate_plan(task=task, context=context)

    async def interpret_command(
        self,
        command: str,
    ) -> Dict[str, Any]:
        """Interpret command using the LangChain LLM.
           (Requires a suitable prompt and potentially adapter logic)
        """
        logger.warning("`interpret_command` called, but specific LangChain adaptation might be needed.")
        # Example generic approach (likely needs refinement):
        prompt = f"Interpret the command: {command}. Output a JSON with 'goal', 'entities', 'plan'."
        response = await self.llm_instance.ainvoke(prompt)
        try:
            # Basic JSON extraction/parsing
            content = response.content
            json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                logger.error(f"Could not extract JSON from interpret_command response: {content}")
                raise ValueError("Failed to parse interpret_command response")
        except Exception as e:
            logger.error(f"Error parsing interpret_command response: {e}", exc_info=True)
            raise ValueError(f"Failed to interpret command: {e}")

    async def generate_recovery_plan(
        self,
        error: str,
        action: Dict[str, Any],
        dom_context: str
    ) -> Dict[str, Any]:
        """Generate a recovery plan using LangChain LLM."""
        from nova.core.prompts import ERROR_RECOVERY_PROMPT # Assuming this prompt exists
        chain = ERROR_RECOVERY_PROMPT | self.llm_instance | self._parser
        try:
            result = await chain.ainvoke({
                "error": error,
                "action": action,
                "dom_context": dom_context
            })
            return result
        except Exception as e:
            logger.error(f"Recovery plan generation failed: {e}", exc_info=True)
            raise

    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a summary response using LangChain LLM."""
        # Use the adapter which contains the correct prompt
        adapter = LangChainAdapter(self.llm_instance)
        return await adapter.generate_response(task=task, plan=plan, context=context)

    def get_langchain_llm(self) -> BaseChatModel:
        """Return the Langchain LLM instance."""
        # Instance is always BaseChatModel now
        return self.llm_instance 