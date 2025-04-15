"""Script to run the Nova dashboard server."""

import argparse
from dotenv import load_dotenv  # Import the function
from src.nova.web.server import run_server

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Nova dashboard server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload) 