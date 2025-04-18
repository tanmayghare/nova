"""LangChain browser tools for Nova."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from ...core.browser import Browser

# Input schemas for each tool
class NavigateInput(BaseModel):
    """Input schema for navigate tool."""
    url: str = Field(..., description="URL to navigate to")

class ClickInput(BaseModel):
    """Input schema for click tool."""
    selector: str = Field(..., description="CSS selector for element to click")

class TypeInput(BaseModel):
    """Input schema for type tool."""
    selector: str = Field(..., description="CSS selector for element to type into")
    text: str = Field(..., description="Text to type")

class WaitInput(BaseModel):
    """Input schema for wait tool."""
    selector: str = Field(..., description="CSS selector for element to wait for")
    timeout: Optional[float] = Field(10.0, description="Timeout in seconds")

class GetTextInput(BaseModel):
    """Input schema for get_text tool."""
    selector: str = Field(..., description="CSS selector for element to get text from")

class GetHtmlInput(BaseModel):
    """Input schema for get_html tool."""
    pass  # No input parameters needed

class ScreenshotInput(BaseModel):
    """Input schema for screenshot tool."""
    path: Optional[str] = Field(None, description="Optional path to save screenshot")

class ScrollInput(BaseModel):
    """Input schema for scroll tool."""
    direction: str = Field(..., description="Direction to scroll: 'up', 'down', 'top', or 'bottom'")

# Tool implementations
class NavigateTool(BaseTool):
    """Tool for navigating to a URL."""
    name = "navigate"
    description = "Navigate to a URL"
    args_schema = NavigateInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self, url: str) -> Dict[str, Any]:
        await self.browser.navigate(url)
        return {"status": "success", "url": url}

class ClickTool(BaseTool):
    """Tool for clicking elements."""
    name = "click"
    description = "Click an element"
    args_schema = ClickInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self, selector: str) -> Dict[str, Any]:
        await self.browser.click(selector)
        return {"status": "success", "selector": selector}

class TypeTool(BaseTool):
    """Tool for typing text."""
    name = "type"
    description = "Type text into an element"
    args_schema = TypeInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self, selector: str, text: str) -> Dict[str, Any]:
        await self.browser.type(selector, text)
        return {"status": "success", "selector": selector, "text": text}

class WaitTool(BaseTool):
    """Tool for waiting for elements."""
    name = "wait"
    description = "Wait for an element to appear"
    args_schema = WaitInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self, selector: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        await self.browser.wait(selector, timeout=timeout)
        return {"status": "success", "selector": selector}

class GetTextTool(BaseTool):
    """Tool for getting text from elements."""
    name = "get_text"
    description = "Get text content from an element"
    args_schema = GetTextInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self, selector: str) -> Dict[str, Any]:
        text = await self.browser.get_text(selector)
        return {"status": "success", "text": text, "selector": selector}

class GetHtmlTool(BaseTool):
    """Tool for getting HTML content."""
    name = "get_html"
    description = "Get HTML content of the current page"
    args_schema = GetHtmlInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self) -> Dict[str, Any]:
        html = await self.browser.get_html_source()
        return {"status": "success", "html": html}

class ScreenshotTool(BaseTool):
    """Tool for taking screenshots."""
    name = "screenshot"
    description = "Take a screenshot of the current page"
    args_schema = ScreenshotInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self, path: Optional[str] = None) -> Dict[str, Any]:
        screenshot = await self.browser.screenshot(path=path)
        return {"status": "success", "path": path, "screenshot_data_length": len(screenshot)}

class ScrollTool(BaseTool):
    """Tool for scrolling the page."""
    name = "scroll"
    description = "Scroll the page in a specified direction"
    args_schema = ScrollInput
    browser: Browser

    def __init__(self, browser: Browser):
        super().__init__()
        self.browser = browser

    async def _arun(self, direction: str) -> Dict[str, Any]:
        script = {
            "up": "window.scrollBy(0, -window.innerHeight/2)",
            "down": "window.scrollBy(0, window.innerHeight/2)",
            "top": "window.scrollTo(0, 0)",
            "bottom": "window.scrollTo(0, document.body.scrollHeight)"
        }.get(direction.lower(), "window.scrollBy(0, window.innerHeight/2)")
        
        await self.browser.evaluate(script)
        return {"status": "success", "direction": direction}

def get_browser_tools(browser: Browser) -> list[BaseTool]:
    """Get all browser tools as LangChain BaseTool instances."""
    return [
        NavigateTool(browser),
        ClickTool(browser),
        TypeTool(browser),
        WaitTool(browser),
        GetTextTool(browser),
        GetHtmlTool(browser),
        ScreenshotTool(browser),
        ScrollTool(browser)
    ] 