import asyncio
import logging
import subprocess
from nova.core.agent import Agent
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrowserAutomation:
    def __init__(self):
        self.agent = None
        
    async def setup(self):
        """Setup the browser automation environment."""
        # Create LLM wrapper with NIM provider
        llm = LLM(
            provider="nim",
            model_name="nvidia/llama-3.3-nemotron-super-49b-v1",
            temperature=0.0,  # Lower temperature for more deterministic responses
            max_tokens=4096
        )
        
        # Create configurations
        config = AgentConfig(
            max_steps=10,
            temperature=0.0,  # Also lower temperature in agent config
        )
        
        # Configure browser to run in headless mode
        browser_config = BrowserConfig(
            headless=True,  # Run in headless mode
            timeout=30,
            viewport={"width": 1280, "height": 720}
        )
        
        # Create agent
        self.agent = Agent(
            llm=llm,
            memory=Memory(),
            config=config,
            browser_config=browser_config,
        )
        
        # Start the agent (this also starts the browser pool)
        await self.agent.start()

    async def run_task(self):
        """Run the browser automation task."""
        try:
            # Define a simple task
            task = """
            Navigate to https://example.com
            Get the page title
            Get the text content of the h1 element
            """
            
            print("Executing task in headless mode...")
            result = await self.agent.run(task)
            print("\nTask Result:")
            print(result)
            return result
            
        except Exception as e:
            logger.error("Task execution failed", exc_info=True)
            raise

    async def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        try:
            # Stop the agent if it exists and hasn't been stopped
            if self.agent is not None:
                print("Stopping agent...")
                await self.agent.stop()
                print("Agent stopped.")
                self.agent = None

            # Get all tasks except the current one
            current_task = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current_task]
            
            if tasks:
                print(f"Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
                except asyncio.TimeoutError:
                    print("Some tasks did not complete within timeout")

            # Get the event loop
            loop = asyncio.get_running_loop()
            
            # Clean up subprocesses
            for transport in getattr(loop, '_transports', set()).copy():
                try:
                    if hasattr(transport, '_proc'):
                        proc = transport._proc
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=1.0)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait(timeout=1.0)
                except Exception as e:
                    print(f"Error terminating process: {e}")

            # Close transports after processes are terminated
            for transport in getattr(loop, '_transports', set()).copy():
                try:
                    if hasattr(transport, 'close') and not transport.is_closing():
                        transport.close()
                except Exception as e:
                    print(f"Error closing transport: {e}")

            # Final cleanup
            try:
                await loop.shutdown_asyncgens()
            except Exception as e:
                print(f"Error during asyncgens shutdown: {e}")
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

async def main():
    """Main async function."""
    automation = BrowserAutomation()
    try:
        await automation.setup()
        await automation.run_task()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
    finally:
        await automation.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"Error running script: {e}") 