"""Browser automation tools."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from ...core.browser import Browser

class BrowserToolSchema(BaseModel):
    """Base schema for browser tools."""
    description: str = Field(..., description="Description of what the tool does")
    args_schema: type[BaseModel] = Field(..., description="Schema for tool arguments")

class NavigateToolSchema(BrowserToolSchema):
    """Schema for navigation tool."""
    url: str = Field(..., description="URL to navigate to")

class ClickToolSchema(BrowserToolSchema):
    """Schema for click tool."""
    selector: str = Field(..., description="CSS selector to click")

class TypeToolSchema(BrowserToolSchema):
    """Schema for type tool."""
    selector: str = Field(..., description="CSS selector to type into")
    text: str = Field(..., description="Text to type")

class WaitToolSchema(BrowserToolSchema):
    """Schema for wait tool."""
    selector: str = Field(..., description="CSS selector to wait for")

class ScreenshotToolSchema(BrowserToolSchema):
    """Schema for screenshot tool."""
    path: Optional[str] = Field(None, description="Path to save screenshot")

class BrowserTool(BaseTool):
    """Base class for browser tools."""
    browser: Browser
    schema: BrowserToolSchema

    def __init__(self, browser: Browser, schema: BrowserToolSchema):
        super().__init__()
        self.browser = browser
        self.schema = schema
        self.name = schema.__class__.__name__.replace("Schema", "")
        self.description = schema.description
        self.args_schema = schema.args_schema

class NavigateTool(BrowserTool):
    """Tool for navigating to a URL."""
    def __init__(self, browser: Browser):
        super().__init__(browser, NavigateToolSchema(
            description="Navigate to a URL",
            args_schema=NavigateToolSchema
        ))

    async def _arun(self, url: str) -> Dict[str, Any]:
        return await self.browser.navigate(url)

class ClickTool(BrowserTool):
    """Tool for clicking elements."""
    def __init__(self, browser: Browser):
        super().__init__(browser, ClickToolSchema(
            description="Click an element",
            args_schema=ClickToolSchema
        ))

    async def _arun(self, selector: str) -> Dict[str, Any]:
        return await self.browser.click(selector)

class TypeTool(BrowserTool):
    """Tool for typing text."""
    def __init__(self, browser: Browser):
        super().__init__(browser, TypeToolSchema(
            description="Type text into an element",
            args_schema=TypeToolSchema
        ))

    async def _arun(self, selector: str, text: str) -> Dict[str, Any]:
        return await self.browser.type(selector, text)

class WaitTool(BrowserTool):
    """Tool for waiting for elements."""
    def __init__(self, browser: Browser):
        super().__init__(browser, WaitToolSchema(
            description="Wait for an element to appear",
            args_schema=WaitToolSchema
        ))

    async def _arun(self, selector: str) -> Dict[str, Any]:
        return await self.browser.wait_for_selector(selector)

class ScreenshotTool(BrowserTool):
    """Tool for taking screenshots."""
    def __init__(self, browser: Browser):
        super().__init__(browser, ScreenshotToolSchema(
            description="Take a screenshot",
            args_schema=ScreenshotToolSchema
        ))

    async def _arun(self, path: Optional[str] = None) -> Dict[str, Any]:
        return await self.browser.take_screenshot(path)

def get_browser_tools(browser: Browser) -> list[BrowserTool]:
    """Get all browser tools."""
    return [
        NavigateTool(browser),
        ClickTool(browser),
        TypeTool(browser),
        WaitTool(browser),
        ScreenshotTool(browser)
    ] 