"""Mock browser for testing."""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

from nova.core.browser import Browser

class MockBrowser(Browser):
    """Mock browser for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mock browser."""
        super().__init__(config or {})
        
        # Mock methods
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.navigate = AsyncMock(return_value={"status": "success"})
        self.click = AsyncMock(return_value={"status": "success"})
        self.type = AsyncMock(return_value={"status": "success"})
        self.get_text = AsyncMock(return_value={"status": "success", "text": "Mock text"})
        self.get_html = AsyncMock(return_value={"status": "success", "html": "<html>Mock HTML</html>"})
        self.screenshot = AsyncMock(return_value={"status": "success", "path": "mock_screenshot.png"})
        self.wait_for_selector = AsyncMock(return_value={"status": "success"})
        self.scroll = AsyncMock(return_value={"status": "success"})
        
        # Mock page
        self.page = MagicMock()
        self.page.url = "https://example.com"
        self.page.title = "Mock Page"
        
    async def start(self) -> None:
        """Mock start method."""
        await self.start()
        
    async def stop(self) -> None:
        """Mock stop method."""
        await self.stop()
        
    async def navigate(self, url: str) -> Dict[str, Any]:
        """Mock navigate method."""
        return await self.navigate(url)
        
    async def click(self, selector: str) -> Dict[str, Any]:
        """Mock click method."""
        return await self.click(selector)
        
    async def type(self, selector: str, text: str) -> Dict[str, Any]:
        """Mock type method."""
        return await self.type(selector, text)
        
    async def get_text(self, selector: str) -> Dict[str, Any]:
        """Mock get_text method."""
        return await self.get_text(selector)
        
    async def get_html(self, selector: str) -> Dict[str, Any]:
        """Mock get_html method."""
        return await self.get_html(selector)
        
    async def screenshot(self, path: str) -> Dict[str, Any]:
        """Mock screenshot method."""
        return await self.screenshot(path)
        
    async def wait_for_selector(self, selector: str, timeout: int = 30) -> Dict[str, Any]:
        """Mock wait_for_selector method."""
        return await self.wait_for_selector(selector, timeout)
        
    async def scroll(self, selector: str) -> Dict[str, Any]:
        """Mock scroll method."""
        return await self.scroll(selector) 