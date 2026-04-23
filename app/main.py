import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from routes import search, bulk_search

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


def validate_environment():
    """Validate required environment variables at startup."""
    required = ["OPENROUTER_API_KEY"]
    missing = []
    
    for var in required:
        value = os.getenv(var)
        if not value or value == "":
            missing.append(var)
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    
    searxng_url = os.getenv("SEARXNG_BASE_URL", "https://searxngapp.app.digitalgalaxy.com:8080")
    logger.info(f"SearXNG URL: {searxng_url}")
    logger.info(f"OpenRouter API Key: {'*' * 20}{os.getenv('OPENROUTER_API_KEY', '')[-4:]}")

    mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
    logger.info(f"Mock Mode: {mock_mode}")
    
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Person Name Finder API...")
    if not validate_environment():
        logger.warning("Running in limited mode - some features may not work")
    yield
    logger.info("API server shutting down")


app = FastAPI(
    title="Person Name Finder API",
    description="Find the current holder of a designation at any company using AI and SearXNG.",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return _rate_limit_exceeded_handler(request, exc)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, tags=["Single Search"])
app.include_router(bulk_search.router, tags=["Bulk Search"])

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Person Name Finder API (Headless)",
        "endpoints": {
            "search": "/search (POST)",
            "bulk-search": "/bulk-search (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
