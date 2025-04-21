"""Browser tools for Nova."""

import logging
from typing import Dict, Optional
from .browser import Browser
from ..tools import Tool, ToolResult
from ..tools.tools import ToolConfig
import base64

logger = logging.getLogger(__name__)

class BrowserTools(Tool):
    """Browser tools for web automation."""
    
    def __init__(self, browser: Browser):
        """Initialize browser tools.
        
        Args:
            browser: Browser instance to use
        """
        # Initialize with a ToolConfig for the suite itself (can be minimal)
        suite_config = ToolConfig(
            name="browser_tools",
            description="Tools for interacting with a web browser.",
            input_schema={},
            output_schema={}
        )
        super().__init__(suite_config)
        self.browser = browser
        
        # Define sub-tools with their specific configurations
        self.tools: Dict[str, ToolConfig] = {
            "navigate": ToolConfig(
                name="navigate",
                description="Navigate the browser to a specific URL.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to navigate to."}
                    },
                    "required": ["url"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Navigation status ('success' or 'error')."},
                        "url": {"type": "string", "description": "The final URL after navigation."},
                        "error": {"type": ["string", "null"], "description": "Error message if navigation failed."}
                    },
                    "required": ["status", "url"]
                },
                func=self.navigate,
                is_async=True
            ),
            "click": ToolConfig(
                name="click",
                description="Click on an element specified by a CSS selector.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector of the element to click."}
                    },
                    "required": ["selector"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Click status ('success' or 'error')."},
                        "selector": {"type": "string", "description": "The selector used."},
                        "error": {"type": ["string", "null"], "description": "Error message if click failed."}
                    }
                },
                func=self.click,
                is_async=True
            ),
            "type": ToolConfig(
                name="type",
                description="Type text into an element specified by a CSS selector. Optionally submit the form.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector of the input element."},
                        "text": {"type": "string", "description": "The text to type."},
                        "submit": {"type": "boolean", "description": "Whether to press Enter after typing.", "default": False}
                    },
                    "required": ["selector", "text"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Type action status ('success' or 'error')."},
                        "selector": {"type": "string", "description": "The selector used."},
                        "text_typed": {"type": "string", "description": "The text that was typed."},
                        "submitted": {"type": "boolean", "description": "If Enter was pressed."},
                        "error": {"type": ["string", "null"], "description": "Error message if typing failed."}
                    }
                },
                func=self.type,
                is_async=True
            ),
            "get_text": ToolConfig(
                name="get_text",
                description="Get the text content of an element specified by a CSS selector.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector of the element."}
                    },
                    "required": ["selector"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Get text status ('success' or 'error')."},
                        "selector": {"type": "string", "description": "The selector used."},
                        "text": {"type": ["string", "null"], "description": "The extracted text content."},
                        "error": {"type": ["string", "null"], "description": "Error message if failed."}
                    }
                },
                func=self.get_text,
                is_async=True
            ),
            "wait": ToolConfig(
                name="wait",
                description="Wait for an element specified by a CSS selector to appear on the page.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector of the element to wait for."},
                        "timeout": {"type": "number", "description": "Maximum time to wait in seconds.", "default": 10.0}
                    },
                    "required": ["selector"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Wait status ('success' or 'error')."},
                        "selector": {"type": "string", "description": "The selector waited for."},
                        "error": {"type": ["string", "null"], "description": "Error message if timeout or other error occurred."}
                    }
                },
                func=self.wait,
                is_async=True
            ),
            "screenshot": ToolConfig(
                name="screenshot",
                description="Take a screenshot of the current page. Returns bytes if no path is provided.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": ["string", "null"], "description": "Optional file path to save the screenshot. If null, returns bytes.", "default": None}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Screenshot status ('success' or 'error')."},
                        "path": {"type": ["string", "null"], "description": "Path where screenshot was saved, if provided."},
                        "screenshot_bytes": {"type": ["string", "null"], "description": "Base64 encoded screenshot bytes, if path was null."},
                        "error": {"type": ["string", "null"], "description": "Error message if failed."}
                    }
                },
                func=self.screenshot,
                is_async=True
            ),
            "get_dom_snapshot": ToolConfig(
                name="get_dom_snapshot",
                description="Get a snapshot of the current page's full DOM structure.",
                input_schema={
                    "type": "object",
                    "properties": {}
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "DOM snapshot status ('success' or 'error')."},
                        "dom_snapshot": {"type": ["string", "null"], "description": "The full HTML source of the page."},
                        "error": {"type": ["string", "null"], "description": "Error message if failed."}
                    }
                },
                func=self.get_dom_snapshot,
                is_async=True
            )
        }
        
    def get_tool_configs(self) -> Dict[str, ToolConfig]:
        """Returns the dictionary of individual tool configurations."""
        # The configurations are already defined in self.tools in __init__
        # Simply return the existing dictionary.
        return self.tools
        
    async def navigate(self, url: str) -> ToolResult:
        """Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            ToolResult indicating success or failure
        """
        logger.info(f"Attempting navigation to: {url}")
        try:
            await self.browser.navigate(url)
            final_url = await self.browser.get_current_url()
            return ToolResult(success=True, data={"status": "success", "url": final_url})
        except Exception as e:
            logger.error(f"Navigation failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), data={"status": "error", "url": url})
            
    async def click(self, selector: str) -> ToolResult:
        """Click an element.
        
        Args:
            selector: CSS selector for element
            
        Returns:
            ToolResult indicating success or failure
        """
        logger.info(f"Attempting click on selector: {selector}")
        try:
            await self.browser.click(selector)
            return ToolResult(success=True, data={"status": "success", "selector": selector})
        except Exception as e:
            logger.error(f"Click failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), data={"status": "error", "selector": selector})
            
    async def type(self, selector: str, text: str, submit: bool = False) -> ToolResult:
        """Type text into an element.
        
        Args:
            selector: CSS selector for element
            text: Text to type
            submit: Whether to press Enter after typing
            
        Returns:
            ToolResult indicating success or failure
        """
        logger.info(f"Attempting to type '{text}' into selector: {selector} (submit: {submit})")
        try:
            result_dict = await self.browser.type(selector, text, submit)
            if result_dict.get("success"):
                return ToolResult(success=True, data={
                    "status": "success", 
                    "selector": selector, 
                    "text_typed": text, 
                    "submitted": submit
                })
            else:
                error = result_dict.get("error", "Typing failed for unknown reason.")
                logger.error(f"Type failed internally: {error}")
                return ToolResult(success=False, error=error, data={
                    "status": "error", 
                    "selector": selector, 
                    "error": error
                })
        except Exception as e:
            logger.error(f"Type action failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), data={
                "status": "error", 
                "selector": selector, 
                "error": str(e)
            })
            
    async def wait(self, selector: str, timeout: float = 10.0) -> ToolResult:
        """Wait for an element to appear.

        Args:
            selector: CSS selector for the element.
            timeout: Maximum time to wait in seconds.

        Returns:
            ToolResult indicating success or failure.
        """
        logger.info(f"Waiting for selector: {selector} (timeout: {timeout}s)")
        try:
            await self.browser.wait(selector, timeout)
            return ToolResult(success=True, data={"status": "success", "selector": selector})
        except Exception as e:
            logger.error(f"Wait failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), data={"status": "error", "selector": selector})

    async def screenshot(self, path: Optional[str] = None) -> ToolResult:
        """Take a screenshot of the current page.

        Args:
            path: Optional path to save the screenshot. If None, returns binary data.

        Returns:
            ToolResult containing screenshot bytes (if path is None) or path (if path is provided).
        """
        logger.info(f"Attempting screenshot (path: {path})")
        try:
            screenshot_bytes = await self.browser.screenshot(path=path)
            data = {"status": "success", "path": path}
            if path is None:
                data["screenshot_bytes"] = base64.b64encode(screenshot_bytes).decode() if screenshot_bytes else None
            return ToolResult(success=True, data=data)
        except Exception as e:
            logger.error(f"Screenshot failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), data={"status": "error", "path": path})

    async def get_text(self, selector: str) -> ToolResult:
        """Extract text content from an element.

        Args:
            selector: CSS selector for the element.

        Returns:
            ToolResult containing the extracted text on success.
        """
        logger.info(f"Attempting to get text from selector: {selector}")
        try:
            text = await self.browser.get_text(selector)
            return ToolResult(success=True, data={"status": "success", "selector": selector, "text": text})
        except Exception as e:
            logger.error(f"Get text failed: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), data={"status": "error", "selector": selector})

    async def get_dom_snapshot(self) -> ToolResult:
        """Get a snapshot of the current DOM structure."""
        logger.info("Attempting to get DOM snapshot")
        try:
            # Use get_html_source() instead of get_content()
            dom_snapshot = await self.browser.get_html_source()
            if dom_snapshot is not None:
                 return ToolResult(success=True, data={"status": "success", "dom_snapshot": dom_snapshot})
            else:
                 # Handle case where get_html_source might return None or empty string on failure
                 logger.error("get_html_source returned None or empty string")
                 return ToolResult(success=False, error="Failed to retrieve HTML source", data={"status": "error"})
        except Exception as e:
            logger.error(f"Error getting DOM snapshot: {e}", exc_info=True)
            return ToolResult(success=False, error=str(e), data={"status": "error"}) 