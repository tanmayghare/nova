"""Main CLI entrypoint for Nova."""

import argparse
import asyncio
import logging
import sys
from typing import List, Optional

from nova.cli.commands import (
    run_command,
    info_command,
    list_tools_command,
    interactive_command,
    web_command,
)


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Nova - An intelligent browser automation agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run a task with Nova agent"
    )
    run_parser.add_argument(
        "task",
        help="Task description to execute"
    )
    run_parser.add_argument(
        "--model",
        default="mistral-small3.1:24b-instruct-2503-q4_K_M",
        help="LLM model to use for the agent"
    )
    run_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode"
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start interactive Nova session"
    )
    interactive_parser.add_argument(
        "--model",
        default="mistral-small3.1:24b-instruct-2503-q4_K_M",
        help="LLM model to use for the agent"
    )
    
    # Info command
    subparsers.add_parser(
        "info", help="Display Nova information"
    )
    
    # Tools command
    subparsers.add_parser(
        "tools", help="List available tools"
    )
    
    # Web command
    web_parser = subparsers.add_parser(
        "web", help="Start Nova web dashboard"
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    web_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode)"
    )
    
    return parser


async def main_async(args: Optional[List[str]] = None) -> int:
    """Asynchronous main entry point.
    
    Args:
        args: Optional list of command-line arguments
        
    Returns:
        Exit code
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 0
    
    setup_logging(parsed_args.verbose)
    
    try:
        if parsed_args.command == "run":
            return await run_command(
                task=parsed_args.task,
                model=parsed_args.model,
                headless=parsed_args.headless,
            )
        elif parsed_args.command == "interactive":
            return await interactive_command(
                model=parsed_args.model,
            )
        elif parsed_args.command == "info":
            return await info_command()
        elif parsed_args.command == "tools":
            return await list_tools_command()
        elif parsed_args.command == "web":
            return await web_command(
                host=parsed_args.host,
                port=parsed_args.port,
                reload=parsed_args.reload,
            )
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point.
    
    Args:
        args: Optional list of command-line arguments
        
    Returns:
        Exit code
    """
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main()) 