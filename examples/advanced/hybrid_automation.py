"""Example demonstrating hybrid automation with Nova, combining HTML parsing and vision analysis."""

import os
import asyncio
import logging

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from nova.core.agent import Agent
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.llama import LlamaModel
from nova.core.task_analyzer import TaskAnalyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting hybrid automation script")
    
    # Initialize rich console for better visualization
    console = Console()
    logger.info("Initialized console")
    
    # Configure the agent
    config = AgentConfig(
        browser_config=BrowserConfig(
            headless=False,  # Show browser for demonstration
            viewport={"width": os.environ.get("BROWSER_VIEWPORT_WIDTH"), "height": os.environ.get("BROWSER_VIEWPORT_HEIGHT")},
            highlight_elements=True,
            slow_motion=0,
            browser_args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials"
            ]
        )
    )
    logger.info("Created agent config")
    
    # Initialize Llama model
    logger.info("Initializing Llama model")
    llama_model = LlamaModel(
        model_name=os.environ.get("MODEL_NAME"),
        temperature=os.environ.get("MODEL_TEMPERATURE"),
        max_tokens=os.environ.get("MODEL_MAX_TOKENS"),
        top_p=os.environ.get("MODEL_TOP_P"),
        top_k=os.environ.get("MODEL_TOP_K"),
        repeat_penalty=os.environ.get("MODEL_REPETITION_PENALTY")
    )
    logger.info("Initialized Llama model")
    
    # Create LLM wrapper
    logger.info("Creating LLM wrapper")
    llm = LLM(model=llama_model, model_type="llama")
    
    # Create agent
    logger.info("Creating agent")
    agent = Agent(
        llm=llm,
        memory=Memory(),
        config=config
    )
    
    try:
        # Start the agent
        logger.info("Starting agent")
        await agent.start()
        logger.info("Agent started successfully")
        
        # Example 1: HTML-based task (Product Information Extraction)
        html_task = """
        Go to https://www.amazon.com/s?k=laptop
        Extract the following information from the first 3 products:
        - Product name
        - Price
        - Rating
        - Number of reviews
        Present the information in a structured format.
        """
        
        logger.info("Starting HTML task")
        console.print(Panel(html_task, title="HTML Task", border_style="blue"))
        analysis = TaskAnalyzer.analyze_task(html_task)
        logger.info(f"Task analysis complete: {analysis['approach'].value}")
        console.print(f"Task Analysis: {analysis['approach'].value} approach")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing HTML task...", total=None)
            logger.info("Executing HTML task")
            result = await agent.run(html_task)
            progress.update(task, completed=True)
            logger.info("HTML task completed")
        
        console.print("\nHTML Task Results:")
        console.print(result)
        
        # Wait for user input before next task
        input("\nPress Enter to continue to the next task...")
        
        # Example 2: Visual-based task (Layout Verification)
        visual_task = """
        Go to https://www.amazon.com
        Verify the following visual elements:
        1. The Amazon logo is visible and properly positioned
        2. The search bar is centered and has the correct styling
        3. The navigation menu is properly aligned
        4. Take a screenshot of the page
        """
        
        logger.info("Starting visual task")
        console.print(Panel(visual_task, title="Visual Task", border_style="green"))
        analysis = TaskAnalyzer.analyze_task(visual_task)
        logger.info(f"Task analysis complete: {analysis['approach'].value}")
        console.print(f"Task Analysis: {analysis['approach'].value} approach")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing visual task...", total=None)
            logger.info("Executing visual task")
            result = await agent.run(visual_task)
            progress.update(task, completed=True)
            logger.info("Visual task completed")
        
        console.print("\nVisual Task Results:")
        console.print(result)
        
        # Wait for user input before next task
        input("\nPress Enter to continue to the next task...")
        
        # Example 3: Hybrid task (Product Analysis with Visual Verification)
        hybrid_task = """
        Go to https://www.amazon.com/s?k=laptop
        Perform the following tasks:
        1. Extract product information (name, price, rating) from the first 3 products
        2. Verify that the product images are properly displayed
        3. Check if the "Add to Cart" buttons are visible and properly styled
        4. Take a screenshot of the product listing
        5. Compare the layout with the expected design
        Present the results in a structured format.
        """
        
        logger.info("Starting hybrid task")
        console.print(Panel(hybrid_task, title="Hybrid Task", border_style="yellow"))
        analysis = TaskAnalyzer.analyze_task(hybrid_task)
        logger.info(f"Task analysis complete: {analysis['approach'].value}")
        console.print(f"Task Analysis: {analysis['approach'].value} approach")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing hybrid task...", total=None)
            logger.info("Executing hybrid task")
            result = await agent.run(hybrid_task)
            progress.update(task, completed=True)
            logger.info("Hybrid task completed")
        
        console.print("\nHybrid Task Results:")
        console.print(result)
        
        # Wait for user input before closing
        input("\nPress Enter to close the browser...")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        console.print(f"[red]Error: {str(e)}[/red]")
    finally:
        logger.info("Cleaning up agent")
        await agent.cleanup()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    logger.info("Starting script")
    asyncio.run(main())
    logger.info("Script completed") 