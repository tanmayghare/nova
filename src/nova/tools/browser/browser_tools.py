import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

from nova.core.browser import Browser
from nova.core.tools import ToolConfig

logger = logging.getLogger(__name__)

# --- Tool Schemas ---

class NavigateInput(BaseModel):
    url: str = Field(..., description="The URL to navigate to.")

class NavigateOutput(BaseModel):
    status: str = Field(..., description="Status of the navigation ('success' or 'error').")
    message: str = Field(..., description="A message indicating the result or error.")
    current_url: Optional[str] = Field(None, description="The URL after navigation attempt.")

class ClickInput(BaseModel):
    selector: str = Field(..., description="CSS or XPath selector for the element to click.")

class ClickOutput(BaseModel):
    status: str = Field(..., description="Status of the click operation ('success' or 'error').")
    message: str = Field(..., description="A message indicating the result or error.")

class TypeInput(BaseModel):
    selector: str = Field(..., description="CSS or XPath selector for the input element.")
    text: str = Field(..., description="The text to type into the element.")

class TypeOutput(BaseModel):
    status: str = Field(..., description="Status of the type operation ('success' or 'error').")
    message: str = Field(..., description="A message indicating the result or error.")

class ScreenshotInput(BaseModel):
    path: str = Field(..., description="The file path to save the screenshot.")

class ScreenshotOutput(BaseModel):
    status: str = Field(..., description="Status of the screenshot operation ('success' or 'error').")
    message: str = Field(..., description="A message indicating the result or error.")
    path: Optional[str] = Field(None, description="The actual path where the screenshot was saved.")

class GetDOMInput(BaseModel):
    pass # No input required

class GetDOMOutput(BaseModel):
    status: str = Field(..., description="Status of the DOM retrieval ('success' or 'error').")
    dom: Optional[str] = Field(None, description="The HTML content of the current page.")
    message: str = Field(..., description="A message indicating the result or error.")

class WaitForSelectorInput(BaseModel):
    selector: str = Field(..., description="CSS or XPath selector for the element to wait for.")
    timeout: Optional[float] = Field(10.0, description="Maximum time to wait in seconds.") # Use float for timeout
    state: Optional[str] = Field("visible", description="State to wait for: 'attached', 'detached', 'visible', 'hidden'.")

    @validator('state')
    def validate_state(cls, v):
        if v not in ["attached", "detached", "visible", "hidden"]:
            raise ValueError("State must be one of 'attached', 'detached', 'visible', 'hidden'")
        return v

class WaitForSelectorOutput(BaseModel):
    status: str = Field(..., description="Status of the wait operation ('success' or 'error').")
    message: str = Field(..., description="A message indicating the result or error.")

class GetTextInput(BaseModel):
    selector: str = Field(..., description="CSS or XPath selector for the element to get text from.")

class GetTextOutput(BaseModel):
    status: str = Field(..., description="Status of the get text operation ('success' or 'error').")
    text: Optional[str] = Field(None, description="The extracted text content.")
    message: str = Field(..., description="A message indicating the result or error.")

class ScrollInput(BaseModel):
    direction: str = Field(..., description="Direction to scroll: 'up', 'down', 'top', 'bottom'.")
    # Optional: Add selector to scroll element into view, or pixels to scroll by
    # selector: Optional[str] = Field(None, description="CSS or XPath selector of element to scroll to.")
    # pixels: Optional[int] = Field(None, description="Number of pixels to scroll (positive for down, negative for up).")

    @validator('direction')
    def validate_direction(cls, v):
        if v not in ["up", "down", "top", "bottom"]:
            raise ValueError("Direction must be one of 'up', 'down', 'top', 'bottom'")
        return v

class ScrollOutput(BaseModel):
    status: str = Field(..., description="Status of the scroll operation ('success' or 'error').")
    message: str = Field(..., description="A message indicating the result or error.")

# --- Tool Implementation Functions ---

async def navigate_impl(browser: Browser, url: str) -> Dict[str, Any]:
    """Implementation for the navigate tool."""
    try:
        # Basic URL formatting (can enhance based on old NavigateTool logic)
        url = str(url).strip().strip('"\'`')
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        await browser.navigate(url)
        current_url = browser._page.url if browser._page else None
        logger.info(f"Navigated to {url}")
        return NavigateOutput(status="success", message=f"Navigated to {url}", current_url=current_url).dict()
    except Exception as e:
        logger.error(f"Navigation failed for {url}: {e}", exc_info=True)
        return NavigateOutput(status="error", message=str(e), current_url=browser._page.url if browser._page else None).dict()

async def click_impl(browser: Browser, selector: str) -> Dict[str, Any]:
    """Implementation for the click tool."""
    try:
        await browser.click(selector)
        logger.info(f"Clicked element: {selector}")
        return ClickOutput(status="success", message=f"Clicked element: {selector}").dict()
    except Exception as e:
        logger.error(f"Click failed for selector {selector}: {e}", exc_info=True)
        return ClickOutput(status="error", message=str(e)).dict()

async def type_impl(browser: Browser, selector: str, text: str) -> Dict[str, Any]:
    """Implementation for the type tool."""
    try:
        await browser.type(selector, text)
        logger.info(f"Typed text into {selector}")
        return TypeOutput(status="success", message=f"Typed text into {selector}").dict()
    except Exception as e:
        logger.error(f"Type failed for selector {selector}: {e}", exc_info=True)
        return TypeOutput(status="error", message=str(e)).dict()

async def screenshot_impl(browser: Browser, path: str) -> Dict[str, Any]:
    """Implementation for the screenshot tool."""
    try:
        # Ensure path exists if needed, or handle directory creation
        await browser.screenshot(path=path)
        logger.info(f"Screenshot saved to {path}")
        return ScreenshotOutput(status="success", message=f"Screenshot saved to {path}", path=path).dict()
    except Exception as e:
        logger.error(f"Screenshot failed for path {path}: {e}", exc_info=True)
        return ScreenshotOutput(status="error", message=str(e)).dict()

async def get_dom_impl(browser: Browser) -> Dict[str, Any]:
    """Implementation for the get_dom tool."""
    try:
        dom_content = await browser.get_html()
        logger.info("Retrieved DOM content.")
        return GetDOMOutput(status="success", dom=dom_content, message="Retrieved DOM content.").dict()
    except Exception as e:
        logger.error(f"Failed to retrieve DOM: {e}", exc_info=True)
        return GetDOMOutput(status="error", message=str(e)).dict()

async def wait_for_selector_impl(browser: Browser, selector: str, timeout: float = 10.0, state: str = "visible") -> Dict[str, Any]:
    """Implementation for the wait_for_selector tool."""
    try:
        timeout_ms = int(timeout * 1000) # Convert seconds to milliseconds
        await browser.wait_for_selector(selector, timeout=timeout_ms, state=state)
        logger.info(f"Successfully waited for selector '{selector}' with state '{state}'.")
        return WaitForSelectorOutput(status="success", message=f"Element '{selector}' is now {state}.").dict()
    except Exception as e:
        logger.error(f"Wait failed for selector '{selector}' (state: {state}): {e}", exc_info=True)
        return WaitForSelectorOutput(status="error", message=str(e)).dict()

async def get_text_impl(browser: Browser, selector: str) -> Dict[str, Any]:
    """Implementation for the get_text tool."""
    try:
        text_content = await browser.get_text(selector)
        logger.info(f"Retrieved text from selector '{selector}'.")
        return GetTextOutput(status="success", text=text_content, message=f"Retrieved text from '{selector}'.").dict()
    except Exception as e:
        logger.error(f"Get text failed for selector '{selector}': {e}", exc_info=True)
        return GetTextOutput(status="error", message=str(e)).dict()

async def scroll_impl(browser: Browser, direction: str) -> Dict[str, Any]:
    """Implementation for the scroll tool."""
    try:
        if not browser._page:
             raise RuntimeError("Browser page not available for scrolling.")
             
        scroll_script = ""
        if direction == "down":
            scroll_script = "window.scrollBy(0, window.innerHeight);"
        elif direction == "up":
            scroll_script = "window.scrollBy(0, -window.innerHeight);"
        elif direction == "top":
            scroll_script = "window.scrollTo(0, 0);"
        elif direction == "bottom":
            scroll_script = "window.scrollTo(0, document.body.scrollHeight);"
            
        await browser.evaluate(scroll_script)
        logger.info(f"Scrolled page {direction}.")
        return ScrollOutput(status="success", message=f"Scrolled page {direction}.").dict()
    except Exception as e:
        logger.error(f"Scroll failed (direction: {direction}): {e}", exc_info=True)
        return ScrollOutput(status="error", message=str(e)).dict()

# --- Tool Registration Example --- 

def register_browser_tools(registry: "ToolRegistry", browser: Browser):
    """Registers the core browser tools with a given registry and browser instance."""

    navigate_tool = ToolConfig(
        name="navigate",
        description="Navigate the browser to a specified URL.",
        input_schema=NavigateInput.schema(),
        output_schema=NavigateOutput.schema(),
        func=lambda **kwargs: navigate_impl(browser, **kwargs), # Bind browser instance
        is_async=True
    )
    registry.register(navigate_tool)

    click_tool = ToolConfig(
        name="click",
        description="Click on an element specified by a CSS or XPath selector.",
        input_schema=ClickInput.schema(),
        output_schema=ClickOutput.schema(),
        func=lambda **kwargs: click_impl(browser, **kwargs), # Bind browser instance
        is_async=True
    )
    registry.register(click_tool)

    type_tool = ToolConfig(
        name="type",
        description="Type text into an element specified by a CSS or XPath selector.",
        input_schema=TypeInput.schema(),
        output_schema=TypeOutput.schema(),
        func=lambda **kwargs: type_impl(browser, **kwargs), # Bind browser instance
        is_async=True
    )
    registry.register(type_tool)

    screenshot_tool = ToolConfig(
        name="screenshot",
        description="Take a screenshot of the current browser page and save it to a file.",
        input_schema=ScreenshotInput.schema(),
        output_schema=ScreenshotOutput.schema(),
        func=lambda **kwargs: screenshot_impl(browser, **kwargs), # Bind browser instance
        is_async=True
    )
    registry.register(screenshot_tool)
    
    get_dom_tool = ToolConfig(
        name="get_dom",
        description="Retrieve the full HTML DOM content of the current browser page.",
        input_schema=GetDOMInput.schema(),
        output_schema=GetDOMOutput.schema(),
        func=lambda **kwargs: get_dom_impl(browser, **kwargs), # Bind browser instance
        is_async=True
    )
    registry.register(get_dom_tool)

    # Register wait_for_selector
    wait_tool = ToolConfig(
        name="wait_for_selector",
        description="Wait for a specific element identified by a selector to reach a certain state (e.g., visible, attached) within a timeout.",
        input_schema=WaitForSelectorInput.schema(),
        output_schema=WaitForSelectorOutput.schema(),
        func=lambda **kwargs: wait_for_selector_impl(browser, **kwargs),
        is_async=True
    )
    registry.register(wait_tool)

    # Register get_text
    get_text_tool = ToolConfig(
        name="get_text",
        description="Extract the text content from an element specified by a selector.",
        input_schema=GetTextInput.schema(),
        output_schema=GetTextOutput.schema(),
        func=lambda **kwargs: get_text_impl(browser, **kwargs),
        is_async=True
    )
    registry.register(get_text_tool)
    
    # Register scroll
    scroll_tool = ToolConfig(
        name="scroll",
        description="Scroll the browser window up, down, to the top, or to the bottom of the page.",
        input_schema=ScrollInput.schema(),
        output_schema=ScrollOutput.schema(),
        func=lambda **kwargs: scroll_impl(browser, **kwargs),
        is_async=True
    )
    registry.register(scroll_tool)

    logger.info("Core browser tools registered.")

# Example Usage (Conceptual - would be part of main application logic):
# async def main():
#     from nova.core.tools import ToolRegistry
#     from nova.core.browser import Browser
# 
#     browser = Browser()
#     await browser.start()
# 
#     registry = ToolRegistry()
#     register_browser_tools(registry, browser)
# 
#     # Now the registry can be used by the agent
#     # ... agent logic ...
# 
#     await browser.stop() 