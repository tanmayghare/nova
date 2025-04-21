"""Web Search tool implementation for Nova."""

import logging
from typing import Dict, Any

# Assuming Tool, ToolResult, ToolConfig are accessible, adjust path if needed
from ...core.tools import Tool, ToolResult, ToolConfig 

# Import the LangChain tool
try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    raise ImportError(
        "Could not import langchain_community. Please install it with `pip install langchain-community`."
    )

logger = logging.getLogger(__name__)

class WebSearchTool(Tool):
    """Tool for performing web searches using DuckDuckGo."""

    def __init__(self):
        """Initialize the Web Search tool."""
        config = ToolConfig(
            name="web_search",
            description="Performs a web search using DuckDuckGo to find information online.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string."}
                },
                "required": ["query"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "string", "description": "A string containing the search results snippets."}
                },
                "required": ["results"]
            },
            is_async=True # DuckDuckGoSearchRun supports async
        )
        super().__init__(config)
        # Instantiate the underlying LangChain tool
        self._search_tool = DuckDuckGoSearchRun()

    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the web search for the given query from input_data.
        
        Args:
            input_data: Dictionary containing the 'query' key.
            
        Returns:
            ToolResult containing search results or an error.
        """
        query = input_data.get("query")
        if not query or not isinstance(query, str):
            return ToolResult(success=False, error="Invalid or missing 'query' in input_data")
            
        logger.info(f"Attempting web search for query: '{query}'")
        try:
            # Use the async run method if available, otherwise sync run
            if hasattr(self._search_tool, 'arun'):
                 results = await self._search_tool.arun(query)
            else:
                 results = self._search_tool.run(query)

            logger.info(f"Web search successful for query: '{query}'. Result length: {len(results)}")
            return ToolResult(
                success=True,
                data={"results": results},
                error=None
            )
        except Exception as e:
            logger.error(f"Error during web search for query '{query}': {e}", exc_info=True)
            return ToolResult(
                success=False,
                data={},
                error=f"Failed to perform web search: {str(e)}"
            )

# Helper function to easily get an instance (optional, but can be convenient)
def get_web_search_tool() -> WebSearchTool:
    return WebSearchTool() 