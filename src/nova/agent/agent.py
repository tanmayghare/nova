"""Extended agent implementation with higher-level functionality."""

from __future__ import annotations # Keep this if present

import logging # <-- Import logging
from typing import List, Optional, Dict, Any
from collections import deque
from datetime import datetime
import json

from langchain_core.language_models.chat_models import BaseChatModel

from nova.core.agent import Agent as CoreAgent
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool
from nova.core.browser import Browser

logger = logging.getLogger(__name__) # <-- Get logger instance

class Agent(CoreAgent):
    """Extended agent with additional functionality.
    
    This agent provides a higher-level interface for browser automation and task execution,
    with built-in support for LLM-based decision making and memory management.
    """

    HISTORY_MAX_LENGTH = 5

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Tool]] = None,
        memory: Optional[Memory] = None,
        config: Optional[AgentConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
    ) -> None:
        """Initialize the agent with optional components.
        
        Args:
            llm: Language model for decision making. If None, uses default NIMProvider.
            tools: List of tools available to the agent.
            memory: Memory system for context management.
            config: Agent configuration.
            browser_config: Browser configuration.
        """
        # Determine the LLM to use: passed llm or create a default one
        final_llm = llm
        if final_llm is None:
            final_llm = LLM(
                provider=config.llm_config.provider if config else "nim",
                docker_image=config.llm_config.nim_config.docker_image if config else "nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest",
                api_base=config.llm_config.nim_config.api_base if config else "http://localhost:8000",
                model_name=config.llm_config.model_name if config else "nvidia/llama-3.3-nemotron-super-49b-v1",
                temperature=config.llm_config.temperature if config else 0.2,
                max_tokens=config.llm_config.max_tokens if config else 4096,
                batch_size=config.llm_config.batch_size if config else 4,
                enable_streaming=config.llm_config.enable_streaming if config else True,
            )

        # Initialize core agent
        super().__init__(
            llm=final_llm, # Use the determined LLM
            tools=tools,
            memory=memory,
            config=config,
            browser_config=browser_config,
        )
        self.action_history: deque = deque(maxlen=self.HISTORY_MAX_LENGTH)

    async def _get_structured_dom(self, max_elements: int = 100) -> List[Dict[str, Any]]:
        """Extracts structured information about interactive elements from the current page.

        Uses Playwright to find elements, gathers key attributes, text, and bounding box,
        and returns a list of dictionaries representing these elements.

        Returns:
            A list of dictionaries, where each dictionary contains details of an
            interactive element.
        """
        if not self.browser or not self.browser._page:
            logger.warning("Cannot get structured DOM: Browser or page not available.")
            return []

        page = self.browser._page
        structured_elements = []
        kept_attributes = ['id', 'name', 'class', 'type', 'placeholder', 'aria-label', 'role', 'href', 'alt', 'title']
        logger.debug("Starting DOM extraction...")
        try:
            # --- Test basic page interaction first ---
            try:
                logger.debug("Attempting to get page title...")
                title = await page.title()
                logger.debug(f"Successfully got page title: {title}")
            except Exception as title_e:
                logger.error(f"Failed to get page title: {title_e}", exc_info=True)
                # If we can't even get the title, DOM extraction is likely doomed
                return [] 
            # --- End basic interaction test ---

            # --- Temporarily Simplify Selector for Debugging ---
            selectors = 'a' # Query only for links
            logger.debug(f"Querying simplified selectors: {selectors}")
            # --- End Simplification ---
            element_handles = await page.query_selector_all(selectors)
            logger.debug(f"Found {len(element_handles)} potential interactive elements (using simplified selector).")

            count = 0
            for i, handle in enumerate(element_handles):
                logger.debug(f"Processing element {i+1}/{len(element_handles)}...")
                if count >= max_elements:
                    logger.warning(f"Reached max elements ({max_elements}) for structured DOM.")
                    break
                try:
                    logger.debug(f"  Checking visibility for element {i+1}...")
                    is_visible = await handle.is_visible()
                    logger.debug(f"  Checking enabled status for element {i+1}...")
                    is_enabled = await handle.is_enabled()
                    
                    if not is_visible or not is_enabled:
                        logger.debug(f"  Element {i+1} skipped (visible={is_visible}, enabled={is_enabled})")
                        continue

                    logger.debug(f"  Getting tag name for element {i+1}...")
                    tag_name = (await handle.evaluate('element => element.tagName')).lower()
                    logger.debug(f"  Getting attributes for element {i+1}...")
                    attributes = await handle.evaluate(f'element => {{ const attrs = {{}}; Array.from(element.attributes).forEach(attr => {{ if ({json.dumps(kept_attributes)}.includes(attr.name)) attrs[attr.name] = attr.value; }}); return attrs; }}')
                    logger.debug(f"  Getting text content for element {i+1}...")
                    text_content = (await handle.text_content() or "").strip()
                    logger.debug(f"  Getting bounding box for element {i+1}...")
                    bbox = await handle.bounding_box()
                    
                    # Filter out elements that are not visible, enabled, or have no size
                    if not is_visible or not is_enabled or not bbox or bbox['width'] == 0 or bbox['height'] == 0:
                        continue

                    # Selectively keep useful attributes
                    element_data = {
                        "tag": tag_name,
                        "attributes": {
                            k: v for k, v in attributes.items() if k in kept_attributes
                        },
                        "text": text_content[:150], # Limit text length for brevity
                        "bbox": bbox
                    }

                    logger.debug(f"  Element {i+1} processed successfully.")
                    structured_elements.append(element_data)
                    count += 1

                except Exception as e:
                    logger.warning(f"Error processing element {i+1} for structured DOM: {e}", exc_info=False) 
                finally:
                    pass 

            logger.info(f"Finished DOM extraction. Extracted {len(structured_elements)} elements.")
            return structured_elements

        except Exception as e:
            logger.error(f"Failed during DOM extraction process: {e}", exc_info=True)
            return []

    async def _get_viewport_screenshot_base64(self) -> Optional[str]:
        """Captures a screenshot of the current viewport and returns it as a base64 string.

        Returns:
            A base64 encoded string of the PNG screenshot, or None if capturing fails.
        """
        if not self.browser or not self.browser.page:
            print("Browser/Page not available for screenshot.")
            return None

        page = self.browser.page

        try:
            screenshot_bytes = await page.screenshot(
                # Capture only the viewport, not the full page
                full_page=False,
                type="png", # PNG is generally good for UI elements
            )
            # Encode bytes to base64 string
            import base64
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            return screenshot_base64
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None 

    def _add_action_to_history(
        self,
        action_taken: Dict[str, Any],
        outcome: str, # e.g., "success", "failure"
        error_message: Optional[str] = None
    ) -> None:
        """Adds a record of the executed action and its outcome to the history.

        Args:
            action_taken: Dictionary describing the action performed 
                          (e.g., {'type': 'click', 'selector': '#id'}).
            outcome: String indicating the result (e.g., 'success', 'failure').
            error_message: Optional error details if the action failed.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "action_taken": action_taken,
            "outcome": outcome,
            "error_message": error_message,
        }
        self.action_history.append(record)

    def _get_formatted_history(self, format_type: str = "json") -> str:
        """Formats the action history for inclusion in the LLM prompt.

        Args:
            format_type: The desired format ('json' or 'string').

        Returns:
            A string representation of the action history.
        """
        history_list = list(self.action_history)

        if not history_list:
            return "No actions taken yet."

        if format_type == "json":
            try:
                # Use default=str to handle non-serializable types like datetime if needed, though isoformat is fine
                return json.dumps(history_list, indent=2)
            except TypeError as e:
                print(f"Error serializing history to JSON: {e}")
                # Fallback to string representation
                return self._format_history_as_string(history_list)
        elif format_type == "string":
             return self._format_history_as_string(history_list)
        else:
            print(f"Unsupported history format: {format_type}. Defaulting to string.")
            return self._format_history_as_string(history_list)
            
    def _format_history_as_string(self, history_list: List[Dict[str, Any]]) -> str:
        """Helper to format history as a simple numbered string."""
        formatted_strings = []
        for i, record in enumerate(history_list):
            action_str = ", ".join(f'{k}=\"{v}\"' for k, v in record['action_taken'].items())
            outcome_str = f"Outcome: {record['outcome']}"
            if record['error_message']:
                outcome_str += f" (Error: {record['error_message']})"
            formatted_strings.append(f"{i+1}. Action: [{action_str}] -> {outcome_str}")
        return "\n".join(formatted_strings) 

    async def _execute_browser_action(self, browser: Browser, action: Dict[str, Any]) -> Any:
        # Implementation of _execute_browser_action method
        pass 