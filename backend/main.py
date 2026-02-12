import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import get_chat_service, router

# Load environment variables from .env file
load_dotenv(find_dotenv())


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
# Quiet down noisy libs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def schedule_checkpoint_cleanup():
    """Background task that cleans up old checkpoints once a day."""
    chat_service = get_chat_service()
    while True:
        try:
            # Run cleanup (defaults to 7 days)
            chat_service.cleanup_old_checkpoints(days=7)
        except Exception as e:
            logger.error(f"Failed to run periodic cleanup: {e}")

        # Wait 24 hours
        await asyncio.sleep(24 * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background tasks
    cleanup_task = asyncio.create_task(schedule_checkpoint_cleanup())
    yield
    # Shutdown: Stop background tasks
    cleanup_task.cancel()


app = FastAPI(
    title="Beer RAG API",
    description="Backend API for the Beer Expert RAG-on-Tap system.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Welcome to the Beer RAG API. Visit /docs for documentation."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
