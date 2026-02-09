import logging
import uuid
from functools import lru_cache

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat_service import ChatService
from services.config_service import ConfigService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@lru_cache()
def get_chat_service():
    config = ConfigService()
    return ChatService(config=config)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class RecipeResponse(BaseModel):
    id: str
    name: str
    style: str
    url: str


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest, chat_service: ChatService = Depends(get_chat_service)
):
    """
    Streaming chat endpoint for the RAG-on-Tap AI.
    """
    # Langchain-Postgres requires UUIDs for session_ids
    session_id = request.session_id

    if not session_id or session_id == "default":
        session_id = str(uuid.uuid4())
    else:
        # Validate if it's already a UUID, or hash it into one
        try:
            uuid.UUID(session_id)
        except ValueError:
            # If the user provided a string that's not a UUID,
            # we still hash it to keep the session persistent for that string
            session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id))

    async def event_generator():
        async for chunk in chat_service.astream_chat(
            request.message, session_id=session_id
        ):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/plain")


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
