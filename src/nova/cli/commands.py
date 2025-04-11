"""CLI command implementations."""

import logging
import sys
from typing import Dict, List

from nova import __version__
from ..core.agent import Agent
from ..core.config import BrowserConfig

logger = logging.getLogger(__name__)


async def run_command(task: str, model: str, headless: bool = False) -> int:
    """Run a task with Nova agent.
    
    Args:
        task: Task description to execute
        model: LLM model to use
        headless: Whether to run browser headlessly
        
    Returns:
        Exit code
    """
    try:
        print(f"Running task: {task}")
        
        # Import LlamaModel here to avoid circular imports
        from ..core.llama import LlamaModel
        
        # Create browser config
        browser_config = BrowserConfig(headless=headless)
        
        # Initialize LLM
        llm = LlamaModel(model_name=model)
        
        # Initialize agent
        agent = Agent(
            llm=llm,
            browser_config=browser_config,
        )
        
        # Run the task
        result = await agent.run(task)
        
        print("\nTask Result:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        return 0
    except Exception as e:
        logger.error(f"Error running task: {e}", exc_info=True)
        return 1


async def interactive_command(model: str) -> int:
    """Start interactive Nova session.
    
    Args:
        model: LLM model to use
        
    Returns:
        Exit code
    """
    try:
        print(f"Starting interactive Nova session with model: {model}")
        print("Type 'exit' or 'quit' to end the session\n")
        
        # Import LlamaModel here to avoid circular imports
        from ..core.llama import LlamaModel
        
        # Initialize LLM
        llm = LlamaModel(model_name=model)
        
        # Initialize agent with default config
        agent = Agent(
            llm=llm,
        )
        
        # Start the browser
        await agent.start()
        
        try:
            while True:
                # Get user input
                task = input("\nEnter task (or 'exit'/'quit' to end): ")
                
                # Check for exit command
                if task.lower() in ("exit", "quit"):
                    break
                
                if not task:
                    continue
                
                # Execute task
                print(f"Executing: {task}")
                result = await agent._execute_task(task)
                
                # Display result
                print("\nResult:")
                print("-" * 50)
                print(result)
                print("-" * 50)
        finally:
            # Stop the browser
            await agent.stop()
        
        print("\nExiting interactive session")
        return 0
    except Exception as e:
        logger.error(f"Error in interactive session: {e}", exc_info=True)
        return 1


async def info_command() -> int:
    """Display Nova information.
    
    Returns:
        Exit code
    """
    print(f"Nova version: {__version__}")
    print("\nSystem information:")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    # Try to get Playwright version
    try:
        import importlib.metadata
        version = importlib.metadata.version('playwright')
        print(f"Playwright version: {version}")
    except Exception as e:
        print(f"Playwright: Not installed ({str(e)})")
    
    # Try to get LLM information
    try:
        # Import here to avoid circular imports
        from ..core.llama import LlamaModel
        available_models = await LlamaModel.list_available_models()
        print("\nAvailable LLM models:")
        for model in available_models:
            print(f"  - {model}")
    except Exception as e:
        print(f"\nUnable to retrieve LLM information: {e}")
    
    return 0


async def list_tools_command() -> int:
    """List available tools.
    
    Returns:
        Exit code
    """
    try:
        # Try to use the tool loader, but handle the case when it doesn't work
        try:
            from ..tools.utils.loader import ToolLoader
            
            # Create tool loader
            loader = ToolLoader()
            
            # Discover and load tools
            tool_classes = loader.discover_and_load_tools()
            
            if not tool_classes:
                print("No tools found")
                return 0
            
            # Group tools by category
            tools_by_category: Dict[str, List[Dict]] = {}
            
            for tool_name, tool_class in tool_classes.items():
                config = tool_class.get_default_config()
                category = getattr(config, "category", "Uncategorized")
                
                if category not in tools_by_category:
                    tools_by_category[category] = []
                
                tools_by_category[category].append({
                    "name": config.name,
                    "description": config.description,
                })
            
            # Print tools by category
            print("Available Tools:")
            for category, tools in sorted(tools_by_category.items()):
                print(f"\n{category}:")
                for tool in sorted(tools, key=lambda t: t["name"]):
                    print(f"  - {tool['name']}: {tool['description']}")
        except (ImportError, AttributeError, Exception) as e:
            print("Tool system not fully initialized.")
            print("This feature will be available in a future update.")
            print(f"Error details: {e}")
        
        return 0
    except Exception as e:
        logger.error(f"Error listing tools: {e}", exc_info=True)
        return 1


async def web_command(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> int:
    """Start the Nova web dashboard.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Whether to enable auto-reload on code changes
        
    Returns:
        Exit code
    """
    try:
        print(f"Starting Nova web dashboard on http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            # Try importing FastAPI dependencies
            import uvicorn
            from fastapi import FastAPI
            from fastapi.responses import HTMLResponse
            from fastapi.staticfiles import StaticFiles
            from fastapi.templating import Jinja2Templates
            
            # Import server module - just to check that it can be imported
            from nova.web import server
            
            # Fix: Use a subprocess to run uvicorn instead of calling run_server directly
            import subprocess
            import sys
            
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "nova.web.server:app",
                "--host",
                host,
                "--port",
                str(port),
                "--log-level",
                "debug",
            ]
            
            if reload:
                cmd.append("--reload")
                
            # Run the server in a subprocess
            process = subprocess.Popen(cmd)
            
            # Wait for the server to be stopped
            try:
                process.wait()
            except KeyboardInterrupt:
                # Handle Ctrl+C
                process.terminate()
                print("\nWeb server stopped by user")
            
        except ImportError as e:
            print(f"\nError: {e}")
            print("\nMissing dependencies for web dashboard. Please install required packages:")
            print("pip install fastapi uvicorn jinja2 websockets")
            return 1
        except Exception as e:
            print(f"\nError starting web server: {e}")
            
            # Provide a simple fallback server
            print("\nStarting simple fallback server...")
            
            import http.server
            import socketserver
            
            class SimpleHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    
                    content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Nova Dashboard (Simple Mode)</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: 0 auto; }}
                            h1 {{ color: #4f46e5; }}
                            .info {{ background: #f3f4f6; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                            .version {{ font-size: 0.9em; color: #6b7280; }}
                        </style>
                    </head>
                    <body>
                        <h1>Nova Dashboard (Simple Mode)</h1>
                        <p>The full dashboard could not be loaded due to missing dependencies.</p>
                        
                        <div class="info">
                            <h2>System Information</h2>
                            <p><strong>Nova Version:</strong> {__version__}</p>
                            <p><strong>Python Version:</strong> {sys.version.split()[0]}</p>
                            <p><strong>Platform:</strong> {sys.platform}</p>
                        </div>
                        
                        <p>To use the full dashboard, please install the required dependencies:</p>
                        <pre>pip install fastapi uvicorn jinja2 websockets</pre>
                    </body>
                    </html>
                    """
                    
                    self.wfile.write(content.encode())
            
            with socketserver.TCPServer((host, port), SimpleHandler) as httpd:
                print(f"Simple dashboard available at http://{host}:{port}")
                httpd.serve_forever()
        
        return 0
    except KeyboardInterrupt:
        print("\nWeb server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error starting web server: {e}", exc_info=True)
        return 1 