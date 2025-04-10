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

    def __init__(self, config: BrowserConfig) -> None:
        """Initialize the browser with configuration.
        
        Args:
            config: Browser configuration
        """
        self.config = config
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self) -> None:
        """Start the browser."""
        try:
            self.playwright = await async_playwright().start()
            browser = await self.playwright.chromium.launch(
                headless=self.config.headless,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            self.browser = browser
            self.context = await browser.new_context()
            self.page = await self.context.new_page()
            if self.config.viewport:
                await self.page.set_viewport_size(self.config.viewport)  # type: ignore
        except Exception as e:
            logger.error(f"Failed to start browser: {e}", exc_info=True)
            await self.close()
            raise

    async def stop(self) -> None:
        """Stop the browser."""
        await self.close()

    async def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self.page:
            raise RuntimeError("Browser not started")
        await self.page.goto(url)

    async def get_text(self, selector: str) -> str:
        """Get text content of an element matching the selector."""
        if not self.page:
            raise RuntimeError("Browser not started")
        content = await self.page.text_content(selector)
        if content is None:
            return ""
        return content

    async def get_html(self) -> str:
        """Get the HTML content of the current page."""
        if not self.page:
            raise RuntimeError("Browser not started")
        return await self.page.content()

    async def click(self, selector: str) -> None:
        """Click an element matching the selector."""
        if not self.page:
            raise RuntimeError("Browser not started")
        await self.page.click(selector)

    async def type(self, selector: str, text: str) -> None:
        """Type text into an element matching the selector."""
        if not self.page:
            raise RuntimeError("Browser not started")
        await self.page.fill(selector, text)

    async def get_state(self) -> Dict[str, Any]:
        """Get the current browser state.

        Returns:
            Dict containing the current state
        """
        if not self.page:
            raise RuntimeError("Browser not started")

        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "content": await self.page.content(),
            "viewport": self.config.viewport,
        }

    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        # Track what we've cleaned up to avoid double cleanup
        cleaned = {
            'page': False,
            'context': False,
            'browser': False,
            'playwright': False
        }
        
        try:
            # Clean up page
            if self.page and not cleaned['page']:
                try:
                    await self.page.close()
                    cleaned['page'] = True
                except Exception as e:
                    logger.error(f"Failed to close page: {e}", exc_info=True)
                finally:
                    self.page = None
                
            # Clean up context
            if self.context and not cleaned['context']:
                try:
                    await self.context.close()
                    cleaned['context'] = True
                except Exception as e:
                    logger.error(f"Failed to close context: {e}", exc_info=True)
                finally:
                    self.context = None
                
            # Clean up browser
            if self.browser and not cleaned['browser']:
                try:
                    await self.browser.close()
                    cleaned['browser'] = True
                except Exception as e:
                    logger.error(f"Failed to close browser: {e}", exc_info=True)
                finally:
                    self.browser = None

            # Clean up playwright
            if self.playwright and not cleaned['playwright']:
                try:
                    await self.playwright.stop()
                    cleaned['playwright'] = True
                except Exception as e:
                    logger.error(f"Failed to stop playwright: {e}", exc_info=True)
                finally:
                    self.playwright = None
                    
        except Exception as e:
            logger.error(f"Browser cleanup failed: {e}", exc_info=True)
        finally:
            # Ensure all references are cleared
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None

    async def __aenter__(self) -> "Browser":
        """Context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()
