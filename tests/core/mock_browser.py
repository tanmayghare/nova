"""Mock browser implementation for testing."""

import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock

class MockBrowser:
    """Mock browser implementation for testing."""
    
    def __init__(self):
        """Initialize the mock browser."""
        self._is_open = False
        self._current_url = None
        self._page_content = None
        self._actions = []
        self.page = None
        
    async def start(self) -> None:
        """Start the browser."""
        self._is_open = True
        self.page = MagicMock()
        await asyncio.sleep(0.1)  # Simulate browser startup
        
    async def stop(self) -> None:
        """Stop the browser."""
        self._is_open = False
        self.page = None
        await asyncio.sleep(0.1)  # Simulate browser shutdown
        
    async def open(self) -> None:
        """Open the browser."""
        await self.start()
        
    async def close(self) -> None:
        """Close the browser."""
        await self.stop()
        
    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self._is_open:
            raise RuntimeError("Browser is not open")
            
        self._current_url = url
        self._actions.append({"type": "navigate", "url": url})
        
        # Simulate page load
        await asyncio.sleep(0.2)
        
        # Set mock page content based on URL
        if "amazon.com" in url:
            self._page_content = """
            <div data-component-type="s-search-result">
                <h2>Gaming Laptop 1</h2>
                <span class="a-price">$999.99</span>
            </div>
            <div data-component-type="s-search-result">
                <h2>Gaming Laptop 2</h2>
                <span class="a-price">$1,299.99</span>
            </div>
            <div data-component-type="s-search-result">
                <h2>Gaming Laptop 3</h2>
                <span class="a-price">$899.99</span>
            </div>
            """
        else:
            self._page_content = "<html><body>Mock page content</body></html>"
            
    async def extract(self, selector: str, id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract elements matching the selector."""
        if not self._is_open:
            raise RuntimeError("Browser is not open")
            
        self._actions.append({"type": "extract", "selector": selector, "id": id, "limit": limit})
        
        # Simulate extraction
        await asyncio.sleep(0.1)
        
        # Return mock data
        if "s-search-result" in selector:
            return [
                {"title": "Gaming Laptop 1", "price": "$999.99"},
                {"title": "Gaming Laptop 2", "price": "$1,299.99"},
                {"title": "Gaming Laptop 3", "price": "$899.99"}
            ][:limit]
        else:
            return [{"content": "Mock extracted content"}]
            
    async def click(self, selector: str) -> None:
        """Click an element."""
        if not self._is_open:
            raise RuntimeError("Browser is not open")
            
        self._actions.append({"type": "click", "selector": selector})
        await asyncio.sleep(0.1)
        
    async def type(self, selector: str, text: str) -> None:
        """Type text into an element."""
        if not self._is_open:
            raise RuntimeError("Browser is not open")
            
        self._actions.append({"type": "type", "selector": selector, "text": text})
        await asyncio.sleep(0.1)
        
    def get_actions(self) -> List[Dict[str, Any]]:
        """Get the list of actions performed."""
        return self._actions
        
    def clear_actions(self) -> None:
        """Clear the action history."""
        self._actions = [] 