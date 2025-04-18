"""Browser tools for Nova."""

from typing import Dict, Optional
from .browser import Browser
from ..tools import Tool, ToolResult
from ..tools.tools import ToolConfig

class BrowserTools(Tool):
    """Browser tools for web automation."""
    
    def __init__(self, browser: Browser):
        """Initialize browser tools.
        
        Args:
            browser: Browser instance to use
        """
        config = ToolConfig(
            name="browser_tools",
            description="Tools for browser automation",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"},
                    "selector": {"type": "string", "description": "CSS selector for element"},
                    "text": {"type": "string", "description": "Text to type"},
                    "timeout": {"type": "number", "description": "Timeout in seconds"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "status": {"type": "string"},
                            "message": {"type": "string"}
                        }
                    },
                    "error": {"type": "string"}
                }
            }
        )
        super().__init__(config)
        self.browser = browser
        
        # Register individual tools
        self.tools = {
            "navigate": ToolConfig(
                name="navigate",
                description="Navigate to a URL",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"}
                    },
                    "required": ["url"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "status": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        },
                        "error": {"type": "string"}
                    }
                },
                func=self.navigate,
                is_async=True
            ),
            "click": ToolConfig(
                name="click",
                description="Click an element",
                input_schema={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for element"}
                    },
                    "required": ["selector"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {"type": "object"},
                        "error": {"type": "string"}
                    }
                },
                func=self.click,
                is_async=True
            ),
            "type": ToolConfig(
                name="type",
                description="Type text into an element",
                input_schema={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for element"},
                        "text": {"type": "string", "description": "Text to type"}
                    },
                    "required": ["selector", "text"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "data": {"type": "object"},
                        "error": {"type": "string"}
                    }
                },
                func=self.type,
                is_async=True
            )
        }
        
    def get_tool_configs(self) -> Dict[str, ToolConfig]:
        """Returns the dictionary of individual tool configurations."""
        # Add other browser actions (get_text, wait, screenshot, etc.) here
        # Example for get_text:
        self.tools["get_text"] = ToolConfig(
            name="get_text",
            description="Extract text content from an element",
            input_schema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector for element"}
                },
                "required": ["selector"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {
                        "type": "object", 
                        "properties": {
                             "text": {"type": ["string", "null"]}
                        }
                    },
                    "error": {"type": ["string", "null"]}
                }
            },
            func=self.get_text,
            is_async=True
        )
        # Add wait
        self.tools["wait"] = ToolConfig(
            name="wait",
            description="Wait for an element to appear",
             input_schema={
                 "type": "object",
                 "properties": {
                     "selector": {"type": "string", "description": "CSS selector for element"},
                     "timeout": {"type": "number", "description": "Timeout in seconds", "default": 10.0}
                 },
                 "required": ["selector"]
             },
             output_schema={
                 "type": "object",
                 "properties": {
                     "success": {"type": "boolean"},
                     "data": {"type": "object"}, # Empty data obj on success
                     "error": {"type": ["string", "null"]}
                 }
             },
            func=self.wait,
            is_async=True
        )
        # Add screenshot
        self.tools["screenshot"] = ToolConfig(
            name="screenshot",
            description="Take a screenshot of the current page",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": ["string", "null"], "description": "Optional path to save screenshot"}
                }
                # No required fields
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {
                        "type": "object",
                        "properties": {
                             "screenshot_data_length": {"type": "integer"}, # Indicate data presence
                             "path": {"type": ["string", "null"]}
                        }
                    },
                    "error": {"type": ["string", "null"]}
                }
            },
            func=self.screenshot,
            is_async=True
        )
        # Add get_dom_snapshot 
        self.tools["get_dom_snapshot"] = ToolConfig(
            name="get_dom_snapshot",
            description="Get a snapshot of the current DOM structure",
            input_schema={"type": "object", "properties": {}}, # No input args
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "dom_snapshot": {"type": "string"}
                        }
                    },
                    "error": {"type": ["string", "null"]}
                }
            },
            func=self.get_dom_snapshot,
            is_async=True
        )

        return self.tools
        
    async def navigate(self, url: str) -> ToolResult:
        """Navigate to a URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            ToolResult indicating success or failure
        """
        try:
            await self.browser.navigate(url)
            return ToolResult(
                success=True,
                data={"url": url, "status": "success", "message": f"Navigated to {url}"},
                error=""
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data={"url": url, "status": "error", "message": str(e)},
                error=str(e)
            )
            
    async def click(self, selector: str) -> ToolResult:
        """Click an element.
        
        Args:
            selector: CSS selector for the element
            
        Returns:
            ToolResult indicating success or failure
        """
        try:
            await self.browser.click(selector)
            return ToolResult(success=True, data={"selector": selector})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
            
    async def type(self, selector: str, text: str) -> ToolResult:
        """Type text into an element.
        
        Args:
            selector: CSS selector for the element
            text: Text to type
            
        Returns:
            ToolResult indicating success or failure
        """
        try:
            await self.browser.type(selector, text)
            return ToolResult(success=True, data={"selector": selector, "text": text})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
            
    async def wait(self, selector: str, timeout: Optional[float] = None) -> ToolResult:
        """Wait for an element to appear.
        
        Args:
            selector: CSS selector for the element
            timeout: Maximum time to wait in seconds
            
        Returns:
            ToolResult indicating success or failure
        """
        try:
            await self.browser.wait(selector, timeout=timeout or 10.0)
            return ToolResult(success=True, data={"selector": selector})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
            
    async def screenshot(self, path: Optional[str] = None) -> ToolResult:
        """Take a screenshot.
        
        Args:
            path: Optional path to save the screenshot
            
        Returns:
            ToolResult containing the screenshot data
        """
        try:
            screenshot = await self.browser.screenshot(path=path)
            return ToolResult(success=True, data={"screenshot": screenshot, "path": path})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
            
    async def get_text(self, selector: str) -> ToolResult:
        """Get text from an element.
        
        Args:
            selector: CSS selector for the element
            
        Returns:
            ToolResult containing the text
        """
        try:
            text = await self.browser.get_text(selector)
            return ToolResult(success=True, data={"text": text, "selector": selector})
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))
            
    async def get_dom_snapshot(self) -> ToolResult:
        """Get a snapshot of the current DOM structure.
        
        Returns:
            ToolResult containing the DOM snapshot string
        """
        if not self.browser or not self.browser._page:
            return ToolResult(success=False, data=None, error="Browser or page not available")
            
        try:
            # Get simplified DOM structure
            dom = await self.browser._page.evaluate("""() => {
                function simplifyNode(node) {
                    if (node.nodeType === 3) return node.textContent.trim();
                    if (node.nodeType !== 1) return '';
                    
                    const children = Array.from(node.childNodes)
                        .map(simplifyNode)
                        .filter(Boolean);
                        
                    const attrs = {};
                    for (const attr of node.attributes || []) {
                        if (['id', 'class', 'href', 'src', 'alt', 'title'].includes(attr.name)) {
                            attrs[attr.name] = attr.value;
                        }
                    }
                    
                    return {
                        tag: node.tagName.toLowerCase(),
                        attrs,
                        children: children.length ? children : undefined
                    };
                }
                return JSON.stringify(simplifyNode(document.documentElement), null, 2);
            }""")
            return ToolResult(success=True, data={"dom_snapshot": dom}, error=None)
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Error getting DOM snapshot: {e}") 