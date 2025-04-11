from typing import Any, Dict, Optional

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import Page, ViewportSize, async_playwright

from .config import BrowserConfig


class Browser:
    """Handles browser automation using Playwright."""

    def __init__(self, config: Optional[BrowserConfig] = None):
        """Initialize the browser with configuration."""
        self.config = config or BrowserConfig()
        self._browser: Optional[PlaywrightBrowser] = None
        self._page: Optional[Page] = None

    async def start(self) -> None:
        """Start the browser and create a new page."""
        playwright = await async_playwright().start()
        self._browser = await playwright.chromium.launch(
            headless=self.config.headless, args=self.config.browser_args
        )
        self._page = await self._browser.new_page()
        if self.config.viewport:
            viewport_size = ViewportSize(width=self.config.viewport["width"], height=self.config.viewport["height"])
            await self._page.set_viewport_size(viewport_size)

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

    async def get_text(self, selector: str) -> str:
        """Get text content of an element matching the selector."""
        if not self._page:
            raise RuntimeError("Browser not started")
        content = await self._page.text_content(selector)
        if content is None:
            return ""
        return content

    async def get_html(self) -> str:
        """Get the current page's HTML content."""
        if not self._page:
            raise RuntimeError("Browser not started")
        return await self._page.content()

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> None:
        """Wait for an element matching the selector to appear."""
        if not self._page:
            raise RuntimeError("Browser not started")
        await self._page.wait_for_selector(selector, timeout=timeout)

    async def evaluate(self, script: str) -> Any:
        """Evaluate JavaScript in the page context."""
        if not self._page:
            raise RuntimeError("Browser not started")
        return await self._page.evaluate(script)

    async def screenshot(self, path: str) -> None:
        """Take a screenshot of the current page."""
        if not self._page:
            raise RuntimeError("Browser not started")
        await self._page.screenshot(path=path)

    async def get_cookies(self) -> Dict[str, str]:
        """Get all cookies from the current page."""
        if not self._page:
            raise RuntimeError("Browser not started")
        cookies = await self._page.context.cookies()
        return {cookie["name"]: cookie["value"] for cookie in cookies}

    async def set_cookies(self, cookies: Dict[str, str]) -> None:
        """Set cookies for the current page."""
        if not self._page:
            raise RuntimeError("Browser not started")
        await self._page.context.add_cookies(
            [
                {"name": name, "value": value, "url": self._page.url}
                for name, value in cookies.items()
            ]
        )
