"""
Enhanced browser automation example with Nova, featuring visual feedback.
This example demonstrates:
- Live browser visualization
- Step-by-step execution with delays
- Rich console output
- Element highlighting
"""

import os
import asyncio

from rich.console import Console
from rich.panel import Panel
from nova.core.agent import Agent
from nova.core.config import AgentConfig, BrowserConfig, LlmConfig

async def main():
    # Initialize rich console for visual feedback
    console = Console()
    
    # Configure the agent
    config = AgentConfig(
        llm_config=LlmConfig(
            model_path="path/to/llama/model",
            temperature=0.1,
            max_tokens=2048
        ),
        browser_config=BrowserConfig(
            headless=False,
            viewport={"width": os.environ.get("BROWSER_VIEWPORT_WIDTH"), "height": os.environ.get("BROWSER_VIEWPORT_HEIGHT")},
            highlight_elements=True,
            slow_motion=0
        )
    )
    
    # Create agent
    agent = Agent(config)
    
    try:
        # Start the agent
        await agent.start()
        
        # Example 1: HTML-based task
        html_task = """
        Go to example.com and extract all links from the page.
        """
        console.print(Panel(html_task, title="HTML Task", border_style="blue"))
        await agent.run(html_task)
        
        # Wait for user input before next task
        input("\nPress Enter to continue to the next task...")
        
        # Example 2: Visual-based task
        visual_task = """
        Go to example.com and verify that the main heading is visible and properly styled.
        Take a screenshot of the page.
        """
        console.print(Panel(visual_task, title="Visual Task", border_style="green"))
        await agent.run(visual_task)
        
        # Wait for user input before next task
        input("\nPress Enter to continue to the next task...")
        
        # Example 3: Hybrid task
        hybrid_task = """
        Go to example.com and:
        1. Extract all links from the page
        2. Verify that the main heading is visible
        3. Take a screenshot
        4. Compare the visual layout with expected design
        """
        console.print(Panel(hybrid_task, title="Hybrid Task", border_style="yellow"))
        await agent.run(hybrid_task)
        
        # Wait for user input before closing
        input("\nPress Enter to close the browser...")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 