import json
import logging
from typing import Dict, Any
from datetime import datetime
from urllib.parse import urlparse, urlunparse
from .llm import LLM
from .browser import Browser
from .exceptions import TaskExecutionError
import re

logger = logging.getLogger(__name__)

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

class Task:
    def __init__(self, task_id: str, description: str, model: str = "mistral-small3.1:24b-instruct-2503-q4_K_M", headless: bool = False):
        self.task_id = task_id
        self.description = description
        self.model = model
        self.headless = headless
        self.status = "pending"
        self.created_at = datetime.now().isoformat()
        self.result = None
        self.error = None
        self.llm = LLM(model=model)
        self.browser = Browser(headless=headless)
        self.timeout = 60  # Add timeout for task execution

    async def execute(self) -> Dict[str, Any]:
        """Execute the task with timeout and error handling."""
        try:
            # Start execution timer
            start_time = datetime.now()
            
            # Update status
            self.status = "running"
            logger.info(f"Starting task execution: {self.task_id}")
            
            # Format URLs in the task description
            formatted_description = re.sub(
                r'https?://\S+',
                lambda m: format_url(m.group(0)),
                self.description
            )

            # Generate plan with retries
            logger.info(f"Generating plan for task: {formatted_description}")
            plan_result = await self.llm.generate_plan(formatted_description)
            logger.info(f"Raw plan result: {plan_result}")
            
            if not isinstance(plan_result, dict):
                raise TaskExecutionError(f"Invalid plan result format: {type(plan_result)}")

            # Validate plan result structure
            if "status" not in plan_result or plan_result["status"] != "success":
                error_msg = plan_result.get("error", "Unknown error")
                raise TaskExecutionError(f"Plan generation failed: {error_msg}")
            
            if "plan" not in plan_result:
                raise TaskExecutionError("Invalid plan result: missing plan field")

            plan = plan_result["plan"]
            if not isinstance(plan, dict) or "steps" not in plan:
                raise TaskExecutionError("Invalid plan format: missing steps")

            steps = plan["steps"]
            if not isinstance(steps, list):
                raise TaskExecutionError("Invalid plan format: steps must be a list")

            # Execute each step with timeout
            results = []
            for i, step in enumerate(steps):
                # Check execution time
                if (datetime.now() - start_time).total_seconds() > self.timeout:
                    raise TaskExecutionError("Task execution timeout")
                    
                try:
                    logger.info(f"Executing step {i + 1}: {json.dumps(step)}")
                    step_result = await self._execute_step(step)
                    results.append(step_result)
                    
                    if step_result.get("status") == "error":
                        raise TaskExecutionError(f"Step {i + 1} failed: {step_result.get('error')}")
                        
                except Exception as e:
                    error_msg = f"Step {i + 1} execution failed: {str(e)}"
                    logger.error(error_msg)
                    raise TaskExecutionError(error_msg)

            # Update task status
            self.status = "success"
            self.result = {
                "steps": results,
                "reasoning": plan.get("reasoning")
            }
            logger.info(f"Task completed successfully: {json.dumps(self.result)}")
            
            return self._to_dict()

        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}"
            logger.error(error_msg)
            self.status = "failed"
            self.error = str(e)
            return self._to_dict()

        finally:
            try:
                await self.browser.close()
            except Exception as e:
                logger.error(f"Error closing browser: {str(e)}")

    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step with improved error handling."""
        try:
            if not isinstance(step, dict):
                raise ValueError(f"Invalid step format: {type(step)}")
            
            action = str(step.get("action", "")).lower()
            params = step.get("parameters", {})
            
            if not isinstance(params, dict):
                raise ValueError("Parameters must be a dictionary")

            if not action:
                raise ValueError("Step must have an action")

            if action == "navigate":
                url = str(params.get("url", ""))
                if not url:
                    raise ValueError("URL is required for navigate action")
                formatted_url = format_url(url)
                await self.browser.navigate(formatted_url)
                return {
                    "action": action,
                    "url": formatted_url,
                    "status": "success"
                }

            elif action == "click":
                selector = str(params.get("selector", ""))
                if not selector:
                    raise ValueError("Selector is required for click action")
                await self.browser.click(selector)
                return {
                    "action": action,
                    "selector": selector,
                    "status": "success"
                }

            elif action == "type":
                selector = str(params.get("selector", ""))
                text = str(params.get("text", ""))
                if not selector or not text:
                    raise ValueError("Selector and text are required for type action")
                await self.browser.type(selector, text)
                return {
                    "action": action,
                    "selector": selector,
                    "status": "success"
                }

            else:
                raise ValueError(f"Unsupported action: {action}")

        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            return {
                "action": action if 'action' in locals() else "unknown",
                "status": "error",
                "error": str(e)
            }

    def _to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary with improved error handling."""
        try:
            # Ensure result is properly formatted
            if isinstance(self.result, str):
                try:
                    self.result = json.loads(self.result)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse result as JSON: {str(e)}")
                    self.result = {"error": f"Invalid result format: {str(e)}"}

            # Ensure error is properly formatted
            if isinstance(self.error, str):
                try:
                    self.error = json.loads(self.error)
                except json.JSONDecodeError:
                    # Keep the error as a string if it's not valid JSON
                    pass

            return {
                "task_id": self.task_id,
                "status": self.status,
                "created_at": self.created_at,
                "result": self.result,
                "error": self.error
            }
        except Exception as e:
            logger.error(f"Error converting task to dictionary: {str(e)}")
            return {
                "task_id": self.task_id,
                "status": "error",
                "error": f"Error serializing task: {str(e)}"
            } 