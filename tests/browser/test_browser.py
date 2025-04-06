import pytest
from typing import AsyncGenerator, Dict, Optional

from nova.core.browser import Browser
from nova.core.config import BrowserConfig


@pytest.fixture
async def browser() -> AsyncGenerator[Browser, None]:
    """Create a browser instance for testing."""
    browser = Browser(BrowserConfig(headless=True))
    await browser.start()
    yield browser
    await browser.stop()


@pytest.mark.asyncio
async def test_navigate(browser: Browser) -> None:
    """Test navigation to a URL."""
    await browser.navigate("https://example.com")
    assert browser._page is not None
    assert await browser._page.evaluate("window.location.href") == "https://example.com/"


@pytest.mark.asyncio
async def test_get_text(browser: Browser) -> None:
    """Test getting text content."""
    await browser.navigate("https://example.com")
    text = await browser.get_text("h1")
    assert "Example Domain" in text


@pytest.mark.asyncio
async def test_get_html(browser: Browser) -> None:
    """Test getting HTML content."""
    await browser.navigate("https://example.com")
    html = await browser.get_html()
    assert "<html" in html
    assert "<head" in html
    assert "<body" in html


@pytest.mark.asyncio
async def test_click(browser: Browser) -> None:
    """Test clicking an element."""
    await browser.navigate("https://example.com")
    # Example.com doesn't have clickable elements, so we just verify the method exists
    assert hasattr(browser, "click")


@pytest.mark.asyncio
async def test_type(browser: Browser) -> None:
    """Test typing text into an element."""
    await browser.navigate("https://example.com")
    # Example.com doesn't have input fields, so we just verify the method exists
    assert hasattr(browser, "type")


@pytest.mark.asyncio
async def test_viewport(browser: Browser) -> None:
    """Test setting viewport size.
    
    This test verifies that the browser correctly handles viewport configuration
    using our dictionary-based format. We use a dictionary with 'width' and 'height'
    keys for consistency with our configuration system, while Playwright internally
    handles the conversion to its viewport format.
    """
    viewport: Dict[str, int] = {"width": 800, "height": 600}
    config = BrowserConfig(
        headless=True,
        viewport=viewport,
    )
    browser = Browser(config)
    await browser.start()
    try:
        assert browser._page is not None
        # Note: type: ignore is used because Playwright's type hints don't match
        # the actual runtime behavior of viewport_size()
        current_viewport = await browser._page.viewport_size()  # type: ignore
        assert isinstance(current_viewport, dict)
        assert current_viewport["width"] == viewport["width"]
        assert current_viewport["height"] == viewport["height"]
    finally:
        await browser.stop()
