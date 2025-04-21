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

from ...browser.config import BrowserConfig

logger = logging.getLogger(__name__)


class Browser:
    """A wrapper around Playwright's browser for web automation."""
    
    def __init__(self, config: Optional[BrowserConfig] = None) -> None:
        """Initialize the browser with configuration.
        
        Args:
            config: Browser configuration options
        """
        logger.debug(f"Browser.__init__ received config type: {type(config)}")
        if config:
            logger.debug(f"Browser.__init__ received config dict: {config.model_dump()}")
            logger.debug(f"Browser.__init__ received config has action_timeout: {hasattr(config, 'action_timeout')}")
        else:
            logger.debug("Browser.__init__ received config is None.")
            
        self.config = config or BrowserConfig()
        # Log the state *after* self.config is set
        logger.debug(f"Browser.__init__ self.config set to type: {type(self.config)}")
        logger.debug(f"Browser.__init__ self.config dict: {self.config.model_dump()}")
        logger.debug(f"Browser.__init__ self.config has action_timeout: {hasattr(self.config, 'action_timeout')}")

        # Ensure viewport dimensions are integers *before* they are used
        try:
            self.config.viewport_width = int(self.config.viewport_width)
            self.config.viewport_height = int(self.config.viewport_height)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert viewport dimensions to integers: {e}. Using defaults (1280x720).")
            self.config.viewport_width = 1280
            self.config.viewport_height = 720
            
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
                args=self.config.extra_args
            )
            logger.info("Browser launched. Creating new page...")
            self._page = await self._browser.new_page()
            logger.info("Page created.")
            # Set viewport size using width and height from config
            viewport = {"width": self.config.viewport_width, "height": self.config.viewport_height}
            logger.debug(f"Setting viewport to: {viewport}")
            await self._page.set_viewport_size(viewport)
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
            # Wait for navigation and network idle
            await self._page.goto(url, wait_until='networkidle', timeout=self.config.action_timeout * 1000)
            # Wait for JavaScript to finish executing
            await self._page.wait_for_load_state('domcontentloaded')
            await self._page.wait_for_load_state('networkidle')
            # Get and log the page content for debugging
            content = await self._page.content()
            logger.debug(f"Page content after navigation:\n{content}")
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

    async def type(self, selector: str, text: str, submit: bool = False) -> Dict[str, Any]:
        """Type text into an element matching the selector.
        
        Args:
            selector: CSS selector for the element to type into
            text: Text to type
            submit: Whether to submit the form after typing
            
        Returns:
            Dictionary containing the result of the typing action
        """
        if not self._page:
            return {
                "success": False,
                "data": {"error": "Browser not started", "selector": selector},
                "error": "Browser not started"
            }
            
        logger.info(f"Attempting to type into selector: {selector}")
        try:
            # Wait for the page to be fully loaded
            await self._page.wait_for_load_state('networkidle')
            await asyncio.sleep(1)  # Additional wait for dynamic content
            
            # Try to handle cookie consent if present
            try:
                await self._page.click('button:has-text("Accept")', timeout=5000)
                await asyncio.sleep(0.5)
            except Exception:
                logger.debug("No cookie consent found or already accepted")
            
            # Try multiple selectors if the primary one fails
            selectors_to_try = [
                selector,
                'textarea[name="q"]',  # Google sometimes uses textarea
                'input[name="q"]',  # Google's main search input
                'input[type="text"]',  # Generic text input
                'input[aria-label*="Search"]',  # ARIA label
                'input[title*="Search"]'  # Title attribute
            ]
            
            for current_selector in selectors_to_try:
                try:
                    # Wait for element to be visible
                    await self._page.wait_for_selector(current_selector, state='visible', timeout=5000)
                    logger.debug(f"Found element with selector: {current_selector}")
                    
                    # Try to focus and type
                    await self._page.focus(current_selector)
                    await asyncio.sleep(0.5)
                    await self._page.fill(current_selector, text)
                    
                    if submit:
                        await self._page.keyboard.press('Enter')
                    
                    return {
                        "success": True,
                        "data": {
                            "selector": current_selector,
                            "text": text,
                            "submitted": submit
                        },
                        "error": None
                    }
                except Exception as e:
                    logger.debug(f"Selector {current_selector} failed: {str(e)}")
                    continue
            
            # If we get here, none of the selectors worked
            return {
                "success": False,
                "data": {"error": "Could not find any suitable input element", "selector": selector},
                "error": "Could not find any suitable input element"
            }
            
        except Exception as e:
            logger.error(f"Failed to type into selector '{selector}': {e}", exc_info=True)
            return {
                "success": False,
                "data": {"error": str(e), "selector": selector},
                "error": str(e)
            }
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

    async def get_current_url(self) -> str:
        """Get the current URL of the page.
        
        Returns:
            The current URL as a string
        """
        if not self._page:
            logger.error("Cannot get URL: Browser not started or page not available.")
            return ""
        return self._page.url


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
