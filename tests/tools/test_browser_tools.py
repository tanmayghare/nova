"""Tests for browser tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import ValidationError

from nova.tools.browser.langchain_tools import get_browser_tools

@pytest.fixture
def mock_browser():
    browser = MagicMock()
    browser.navigate = AsyncMock()
    browser.click = AsyncMock()
    browser.type = AsyncMock()
    browser.get_text = AsyncMock(return_value="test text")
    browser.get_html_source = AsyncMock(return_value="<html>test</html>")
    browser.screenshot = AsyncMock(return_value=b"test screenshot")
    browser.wait = AsyncMock()
    browser.evaluate = AsyncMock()
    return browser

@pytest.fixture
def tools(mock_browser):
    return get_browser_tools(mock_browser)

@pytest.mark.asyncio
async def test_navigate_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "navigate")
    result = await tool._arun(url="https://example.com")
    assert result["status"] == "success"
    assert result["url"] == "https://example.com"
    mock_browser.navigate.assert_called_once_with("https://example.com")

@pytest.mark.asyncio
async def test_click_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "click")
    result = await tool._arun(selector="#button")
    assert result["status"] == "success"
    assert result["selector"] == "#button"
    mock_browser.click.assert_called_once_with("#button")

@pytest.mark.asyncio
async def test_type_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "type")
    result = await tool._arun(selector="#input", text="Hello")
    assert result["status"] == "success"
    assert result["selector"] == "#input"
    assert result["text"] == "Hello"
    mock_browser.type.assert_called_once_with("#input", "Hello")

@pytest.mark.asyncio
async def test_get_text_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "get_text")
    result = await tool._arun(selector="#content")
    assert result["status"] == "success"
    assert result["text"] == "test text"
    assert result["selector"] == "#content"
    mock_browser.get_text.assert_called_once_with("#content")

@pytest.mark.asyncio
async def test_get_html_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "get_html")
    result = await tool._arun()
    assert result["status"] == "success"
    assert result["html"] == "<html>test</html>"
    mock_browser.get_html_source.assert_called_once()

@pytest.mark.asyncio
async def test_screenshot_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "screenshot")
    result = await tool._arun(path="test.png")
    assert result["status"] == "success"
    assert result["path"] == "test.png"
    assert result["screenshot_data_length"] == len(b"test screenshot")
    mock_browser.screenshot.assert_called_once_with(path="test.png")

@pytest.mark.asyncio
async def test_wait_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "wait")
    result = await tool._arun(selector="#loading", timeout=5.0)
    assert result["status"] == "success"
    assert result["selector"] == "#loading"
    mock_browser.wait.assert_called_once_with("#loading", timeout=5.0)

@pytest.mark.asyncio
async def test_scroll_tool(tools, mock_browser):
    tool = next(t for t in tools if t.name == "scroll")
    result = await tool._arun(direction="down")
    assert result["status"] == "success"
    assert result["direction"] == "down"
    mock_browser.evaluate.assert_called_once()

def test_tool_validation(tools):
    # Test invalid input for navigate tool
    tool = next(t for t in tools if t.name == "navigate")
    with pytest.raises(ValidationError):
        tool._arun(url="")  # Empty URL should fail validation

    # Test invalid input for click tool
    tool = next(t for t in tools if t.name == "click")
    with pytest.raises(ValidationError):
        tool._arun(selector="")  # Empty selector should fail validation 