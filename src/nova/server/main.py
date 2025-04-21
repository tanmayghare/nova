"""Main FastAPI application for the Nova Agent server."""

import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

# Import API router
from .api import router as api_router

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine base directory for static/template files
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Create FastAPI app instance
app = FastAPI(
    title="Nova Agent Server",
    description="Backend server to manage and interact with the Nova agent.",
    version="0.1.0" # TODO: Link to package version if possible
)

# Mount static files directory
# Ensure the directory exists before mounting
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Set up templates (using Jinja2 for potential future flexibility)
# Serve index.html directly as the root response instead
# templates = Jinja2Templates(directory=STATIC_DIR)

# Configure CORS (Cross-Origin Resource Sharing)
# Explicitly list allowed origins instead of wildcard
origins = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    # Add any other origins your frontend might be served from
] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# Root endpoint - Now serves the main HTML page
@app.get("/", response_class=FileResponse)
async def read_index(request: Request):
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
         # In a real app, return a proper 404 or error page
         return {"error": "index.html not found"} 
    logger.info("Serving index.html")
    return FileResponse(index_path)

# Include the API router
app.include_router(api_router, prefix="/api")

# Placeholder for other routers if needed (e.g., UI specific)
# ...

if __name__ == "__main__":
    logger.info("Starting Nova Agent Server with uvicorn...")
    # Run the server using uvicorn
    # Use reload=True for development to automatically restart on code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 