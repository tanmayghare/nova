"""Mock browser for testing."""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock
from langchain_core.tools import BaseTool

class MockBrowserTool(BaseTool):
    """Mock browser tool for testing."""
    
    name: str = "mock_browser"
    description: str = "A mock browser tool for testing"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mock browser tool."""
        super().__init__()
        self.config = config or {}
        
        # Mock methods
        self._run = AsyncMock(return_value={"status": "success"})
        self._arun = AsyncMock(return_value={"status": "success"})
        
        # Mock page
        self.page = MagicMock()
        self.page.url = "https://example.com"
        self.page.title = "Mock Page"
        
    def _run(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        """Mock run method."""
        return {"status": "success", "action": action, **kwargs}
    
    async def _arun(self, action: str, **kwargs: Any) -> Dict[str, Any]:
        """Mock arun method."""
        return {"status": "success", "action": action, **kwargs}
    
    def navigate(self, url: str) -> Dict[str, Any]:
        """Mock navigate method."""
        return self._run("navigate", url=url)
    
    def click(self, selector: str) -> Dict[str, Any]:
        """Mock click method."""
        return self._run("click", selector=selector)
    
    def type(self, selector: str, text: str) -> Dict[str, Any]:
        """Mock type method."""
        return self._run("type", selector=selector, text=text)
    
    def get_text(self, selector: str) -> Dict[str, Any]:
        """Mock get_text method."""
        return self._run("get_text", selector=selector, text="Mock text")
    
    def get_html(self, selector: str) -> Dict[str, Any]:
        """Mock get_html method."""
        return self._run("get_html", selector=selector, html="<html>Mock HTML</html>")
    
    def screenshot(self, path: str) -> Dict[str, Any]:
        """Mock screenshot method."""
        return self._run("screenshot", path=path, saved_path="mock_screenshot.png")
    
    def wait_for_selector(self, selector: str, timeout: int = 30) -> Dict[str, Any]:
        """Mock wait_for_selector method."""
        return self._run("wait_for_selector", selector=selector, timeout=timeout)
    
    def scroll(self, selector: str) -> Dict[str, Any]:
        """Mock scroll method."""
        return self._run("scroll", selector=selector) 