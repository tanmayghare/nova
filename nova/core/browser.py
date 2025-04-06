from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import (
    BrowserContext,
    Page,
    async_playwright,
)

from .config import BrowserConfig

logger = logging.getLogger(__name__)


class Browser:
    """Handles browser automation using Playwright.
    
    This class provides a high-level interface for browser automation using Playwright.
    It handles viewport configuration using a dictionary format for simplicity and
    consistency with the rest of the configuration system, while internally converting
    to Playwright's viewport format when needed.
    """

    def __init__(self, config: Optional[BrowserConfig] = None) -> None:
        """Initialize the browser with configuration.
        
        Args:
            config: Optional browser configuration. If not provided, uses default config.
                   The viewport configuration should be a dictionary with 'width' and 'height'
                   keys containing integer values.
        """
        self.config = config or BrowserConfig()
        self._browser: Optional[PlaywrightBrowser] = None
        self._page: Optional[Page] = None

    async def start(self) -> None:
        """Start the browser and create a new page.
        
        This method launches the browser with the configured options and creates
        an initial page. If viewport configuration is provided, it sets the viewport
        size for the page.
        """
        playwright = await async_playwright().start()
        self._browser = await playwright.chromium.launch(
            headless=self.config.headless, args=self.config.browser_args
        )
        self._page = await self._browser.new_page()
        if self.config.viewport:
            # Note: We use type: ignore here because Playwright's type hints expect their
            # ViewportSize class, but their implementation actually accepts a compatible dict
            await self._page.set_viewport_size(self.config.viewport)  # type: ignore

    async def stop(self) -> None:
        """Stop the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None

    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self._page:
            raise RuntimeError("Browser not started")
        await self._page.goto(url)

    async def get_text(self, selector: str) -> str:
        """Get text content of an element matching the selector."""
        if not self._page:
            raise RuntimeError("Browser not started")
        content = await self._page.text_content(selector)
        if content is None:
            return ""
        return content

    async def get_html(self) -> str:
        """Get the HTML content of the current page."""
        if not self._page:
            raise RuntimeError("Browser not started")
        return await self._page.content()

    async def click(self, selector: str) -> None:
        """Click an element matching the selector."""
        if not self._page:
            raise RuntimeError("Browser not started")
        await self._page.click(selector)

    async def type(self, selector: str, text: str) -> None:
        """Type text into an element matching the selector."""
        if not self._page:
            raise RuntimeError("Browser not started")
        await self._page.fill(selector, text)

    async def get_state(self) -> Dict[str, Any]:
        """Get the current browser state.

        Returns:
            Dict containing the current state
        """
        if not self._page:
            raise RuntimeError("Browser not started")

        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "content": await self._page.content(),
            "viewport": self.config.viewport,
        }

    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
