"""Browser action tools."""

from typing import Any, Dict, Optional

from nova.core.browser import Browser
from nova.tools.base.tool import BaseTool, ToolConfig, ToolResult


class BrowserActionTool(BaseTool):
    """Base class for browser action tools."""
    
    def __init__(self, browser: Browser, config: Optional[ToolConfig] = None) -> None:
        """Initialize with a browser instance."""
        self.browser = browser
        super().__init__(config)

    @property
    def name(self) -> str:
        """Get the tool name."""
        return self.config.name


class NavigateTool(BrowserActionTool):
    """Tool for browser navigation."""
    
    @classmethod
    def get_default_config(cls) -> ToolConfig:
        return ToolConfig(
            name="navigate",
            description="Navigate to a URL",
            version="1.0.0",
            enabled=True,
            timeout=30,
            retry_attempts=3,
            dependencies=[]
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        try:
            url = input_data.get("url")
            if not url:
                return ToolResult(False, None, "URL is required")
            
            # Robust URL formatting
            # 1. Convert to string in case it's not
            # 2. Handle quotes inside and outside the URL
            # 3. Handle common LLM artifacts like markdown formatting
            url = str(url)
            url = url.strip().strip('"\'')
            # Remove any markdown backticks
            url = url.strip('`')
            
            # Check if URL has proper scheme, add https:// if not
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            
            await self.browser.navigate(url)
            return ToolResult(True, {"status": "success", "message": f"Navigated to {url}"})
        except Exception as e:
            return ToolResult(False, None, str(e))


class ClickTool(BrowserActionTool):
    """Tool for clicking elements."""
    
    @classmethod
    def get_default_config(cls) -> ToolConfig:
        return ToolConfig(
            name="click",
            description="Click an element matching the selector",
            version="1.0.0",
            enabled=True,
            timeout=30,
            retry_attempts=3,
            dependencies=[]
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        try:
            selector = input_data.get("selector")
            if not selector:
                return ToolResult(False, None, "Selector is required")
            
            await self.browser.click(selector)
            return ToolResult(True, f"Clicked element: {selector}")
        except Exception as e:
            return ToolResult(False, None, str(e))


class TypeTool(BrowserActionTool):
    """Tool for typing text."""
    
    @classmethod
    def get_default_config(cls) -> ToolConfig:
        return ToolConfig(
            name="type",
            description="Type text into an element",
            version="1.0.0",
            enabled=True,
            timeout=30,
            retry_attempts=3,
            dependencies=[]
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        try:
            selector = input_data.get("selector")
            text = input_data.get("text")
            if not selector or text is None:
                return ToolResult(False, None, "Selector and text are required")
            
            await self.browser.type(selector, text)
            return ToolResult(True, f"Typed text into {selector}")
        except Exception as e:
            return ToolResult(False, None, str(e))


class WaitTool(BrowserActionTool):
    """Tool for waiting for elements."""
    
    @classmethod
    def get_default_config(cls) -> ToolConfig:
        return ToolConfig(
            name="wait",
            description="Wait for an element to appear",
            version="1.0.0",
            enabled=True,
            timeout=30,
            retry_attempts=3,
            dependencies=[]
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        try:
            selector = input_data.get("selector")
            timeout = input_data.get("timeout", 10.0)
            if not selector:
                return ToolResult(False, None, "Selector is required")
            
            await self.browser.wait(selector, timeout)
            return ToolResult(True, f"Waited for element: {selector}")
        except Exception as e:
            return ToolResult(False, None, str(e))


class ScreenshotTool(BrowserActionTool):
    """Tool for taking screenshots."""
    
    @classmethod
    def get_default_config(cls) -> ToolConfig:
        return ToolConfig(
            name="screenshot",
            description="Take a screenshot of the current page",
            version="1.0.0",
            enabled=True,
            timeout=30,
            retry_attempts=3,
            dependencies=[]
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        try:
            path = input_data.get("path")
            screenshot = await self.browser.screenshot(path)
            return ToolResult(True, {"path": path, "data": screenshot})
        except Exception as e:
            return ToolResult(False, None, str(e)) 