from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import (
    Page,
    async_playwright,
)

from .config import BrowserConfig

logger = logging.getLogger(__name__)


class Browser:
    """A wrapper around Playwright's browser for web automation."""
    
    def __init__(self, config: Optional[BrowserConfig] = None) -> None:
        """Initialize the browser with configuration.
        
        Args:
            config: Browser configuration options
        """
        self.config = config or BrowserConfig()
        self._browser: Optional[PlaywrightBrowser] = None
        self._page: Optional[Page] = None
        self._last_used: Optional[datetime] = None
        
    async def start(self) -> None:
        """Start the browser instance."""
        if self._browser is not None:
            logger.warning("Browser already started")
            return
            
        logger.info("Starting Playwright...")
        try:
            playwright = await async_playwright().start()
            logger.info("Launching browser instance...")
            self._browser = await playwright.chromium.launch(
                headless=self.config.headless,
                args=self.config.browser_args
            )
            logger.info("Browser launched. Creating new page...")
            self._page = await self._browser.new_page()
            logger.info("Page created.")
            if self.config.viewport:
                logger.debug(f"Setting viewport to: {self.config.viewport}")
                await self._page.set_viewport_size(self.config.viewport)
            self._last_used = datetime.now()
            logger.info("Browser startup complete.")
        except Exception as e:
            logger.error(f"Browser startup failed: {e}", exc_info=True)
            # Attempt cleanup if possible
            if self._browser:
                await self._browser.close()
                self._browser = None
                self._page = None
            raise # Re-raise exception after logging and cleanup attempt
        
    async def stop(self) -> None:
        """Stop the browser instance."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
            self._last_used = None
            
    async def navigate(self, url: str) -> None:
        """Navigate to a URL.
        
        Args:
            url: The URL to navigate to
        """
        if not self._page:
            logger.error("Cannot navigate: Browser not started or page not available.")
            raise RuntimeError("Browser not started")
            
        # Ensure URL is properly formatted
        if not isinstance(url, str):
            url = str(url)
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
            
        logger.info(f"Navigating to URL: {url}")
        try:
            logger.debug(f"Calling page.goto('{url}') with default wait/timeout...")
            # Use default wait_until and timeout
            await self._page.goto(url) 
            # --- Add log IMMEDIATELY after await --- 
            logger.debug(f"*** page.goto call HAS RETURNED for {url} ***")
            # --- End Add log ---
            self._last_used = datetime.now()
            logger.info(f"Navigation to {url} successful.")
        except Exception as e:
            logger.error(f"Navigation to {url} failed: {e}", exc_info=True)
            self._last_used = datetime.now() 
            raise 
        
    async def click(self, selector: str) -> None:
        """Click an element matching the selector.
        
        Args:
            selector: CSS selector for the element to click
        """
        if not self._page:
            logger.error("Cannot click: Browser not started or page not available.")
            raise RuntimeError("Browser not started")
        
        try:
            logger.info(f"Attempting standard click on selector: {selector}")
            # Explicitly wait for element to be visible and potentially enabled before clicking
            # Playwright's click auto-waits for visible/enabled, but we add an explicit wait for robustness
            await self._page.wait_for_selector(selector, state='visible', timeout=self.config.action_timeout * 1000)
            logger.debug(f"Element '{selector}' is visible, proceeding with click.")
            
            # Default timeout for click is often 30s in Playwright, adjust if needed
            await self._page.click(selector, timeout=self.config.action_timeout * 1000) # Use configured timeout
            logger.info(f"Standard click successful for selector: {selector}")
        except Exception as e:
            # Check if the error message indicates obstruction (heuristic)
            error_str = str(e).lower()
            if "intercepts pointer events" in error_str or "element is covered by" in error_str:
                logger.warning(f"Standard click failed for selector '{selector}' due to obstruction: {e}. Attempting forced click...")
                try:
                    await self._page.click(selector, force=True, timeout=self.config.action_timeout * 1000)
                    logger.info(f"Forced click successful for selector: {selector}")
                except Exception as force_e:
                    logger.error(f"Forced click also failed for selector '{selector}': {force_e}")
                    raise force_e # Re-raise the error from the forced click
            else:
                # Re-raise original error if it wasn't related to obstruction
                logger.error(f"Standard click failed for selector '{selector}' for reason other than obstruction: {e}")
                raise e
        finally:
            self._last_used = datetime.now()

    async def type(self, selector: str, text: str) -> None:
        """Type text into an element matching the selector.
        
        Args:
            selector: CSS selector for the element to type into
            text: Text to type
        """
        if not self._page:
            raise RuntimeError("Browser not started")
            
        logger.info(f"Attempting to type into selector: {selector}")
        try:
            # Explicitly wait for element to be visible/enabled before filling
            await self._page.wait_for_selector(selector, state='visible', timeout=self.config.action_timeout * 1000)
            logger.debug(f"Element '{selector}' is visible, proceeding with fill.")
            
            # Add timeout to fill for consistency
            await self._page.fill(selector, text, timeout=self.config.action_timeout * 1000) 
            logger.info(f"Successfully typed into selector: {selector}")
        except Exception as e:
            logger.error(f"Failed to type into selector '{selector}': {e}", exc_info=True)
            raise # Re-raise the exception
        finally:
            self._last_used = datetime.now()
        
    async def get_text(self, selector: str) -> str:
        """Get text content from an element matching the selector.
        
        Args:
            selector: CSS selector for the element to get text from
            
        Returns:
            The text content of the element
        """
        if not self._page:
            raise RuntimeError("Browser not started")
        text = await self._page.text_content(selector)
        self._last_used = datetime.now()
        return text
        
    async def wait(self, selector: str, timeout: float = 10.0) -> None:
        """Wait for an element matching the selector to appear.
        
        Args:
            selector: CSS selector for the element to wait for
            timeout: Maximum time to wait in seconds
        """
        if not self._page:
            raise RuntimeError("Browser not started")
        await self._page.wait_for_selector(selector, timeout=timeout * 1000)  # Convert to milliseconds
        self._last_used = datetime.now()
        
    async def screenshot(self, path: Optional[str] = None) -> bytes:
        """Take a screenshot of the current page.
        
        Args:
            path: Optional path to save the screenshot to
            
        Returns:
            The screenshot as bytes
        """
        if not self._page:
            raise RuntimeError("Browser not started")
        screenshot = await self._page.screenshot(path=path)
        self._last_used = datetime.now()
        return screenshot

    async def get_html_source(self) -> str:
        """Get the full HTML source of the current page.
        
        Returns:
            The HTML source code as a string, or an empty string if failed.
        """
        if not self._page:
            logger.error("Cannot get HTML source: Browser not started or page not available.")
            return ""
        
        try:
            logger.debug("Getting page HTML content...")
            content = await self._page.content()
            logger.debug(f"Successfully retrieved HTML content (length: {len(content)}).")
            return content
        except Exception as e:
            logger.error(f"Failed to get page HTML content: {e}", exc_info=True)
            return ""

    def is_expired(self, max_age: timedelta) -> bool:
        """Check if the browser instance has expired.
        
        Args:
            max_age: Maximum age before expiration
            
        Returns:
            True if the browser has expired, False otherwise
        """
        if not self._last_used:
            return False
        return datetime.now() - self._last_used > max_age


class BrowserPool:
    """A pool of browser instances for efficient resource management."""
    
    def __init__(self, size: int = 5, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the browser pool.
        
        Args:
            size: Maximum number of browser instances in the pool
            config: Browser configuration options
        """
        self.size = size
        self.config = config or {}
        self._browsers: List[Browser] = []
        self._available: List[Browser] = []
        self._in_use: List[Browser] = []
        self._lock = asyncio.Lock()
        self._started = False
        self._max_age = timedelta(minutes=30)
        
    @property
    def started(self) -> bool:
        """Check if the pool has been started."""
        return self._started
        
    async def start(self) -> None:
        """Start the browser pool."""
        if self._started:
            logger.warning("Browser pool already started")
            return
            
        async with self._lock:
            # Create initial browser instances
            for _ in range(self.size):
                browser = Browser(self.config)
                await browser.start()
                self._browsers.append(browser)
                self._available.append(browser)
                
            self._started = True
            
    async def stop(self) -> None:
        """Stop the browser pool."""
        if not self._started:
            logger.warning("Browser pool not started")
            return
            
        async with self._lock:
            # Stop all browser instances
            for browser in self._browsers:
                await browser.stop()
                
            self._browsers.clear()
            self._available.clear()
            self._in_use.clear()
            self._started = False
            
    async def acquire(self) -> Browser:
        """Acquire a browser instance from the pool.
        
        Returns:
            An available browser instance
            
        Raises:
            RuntimeError: If no browsers are available
        """
        if not self._started:
            raise RuntimeError("Browser pool not started")
            
        async with self._lock:
            # Check for expired browsers
            for browser in self._available:
                if browser.is_expired(self._max_age):
                    await browser.stop()
                    await browser.start()
                    
            # Get an available browser
            if self._available:
                browser = self._available.pop()
                self._in_use.append(browser)
                return browser
                
            # Create a new browser if possible
            if len(self._browsers) < self.size:
                browser = Browser(self.config)
                await browser.start()
                self._browsers.append(browser)
                self._in_use.append(browser)
                return browser
                
            raise RuntimeError("No browsers available")
            
    async def release(self, browser: Browser) -> None:
        """Release a browser instance back to the pool.
        
        Args:
            browser: The browser instance to release
        """
        if not self._started:
            raise RuntimeError("Browser pool not started")
            
        async with self._lock:
            if browser in self._in_use:
                self._in_use.remove(browser)
                self._available.append(browser)
                
    async def cleanup(self) -> None:
        """Clean up expired browser instances."""
        if not self._started:
            return
            
        async with self._lock:
            # Clean up expired browsers
            for browser in self._available:
                if browser.is_expired(self._max_age):
                    await browser.stop()
                    self._available.remove(browser)
                    self._browsers.remove(browser)
                    
            # Replace cleaned up browsers
            while len(self._browsers) < self.size:
                browser = Browser(self.config)
                await browser.start()
                self._browsers.append(browser)
                self._available.append(browser)
