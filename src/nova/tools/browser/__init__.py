"""Browser automation tools for Nova - Entry point."""

from typing import List
from ...core.browser import Browser # Need Browser type hint
from ...core.browser.tools import BrowserTools # Import the actual tool class

# Define the helper function here directly or import if defined elsewhere
def get_browser_tools(browser: Browser) -> BrowserTools:
    """Instantiates and returns the BrowserTools object."""
    # This assumes BrowserTools constructor takes the browser instance
    return BrowserTools(browser=browser)

__all__ = [
    "get_browser_tools",
    # We don't export BrowserTools itself from here usually,
    # the function is the intended interface.
]
