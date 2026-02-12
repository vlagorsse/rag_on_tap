import logging
import uuid

import psycopg
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from services.config_service import ConfigService, LLMProvider
from services.rag_tool import BeerRAGTool

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        config: ConfigService,
        model_name: str | None = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        rerank_model: str = "Qwen/Qwen3-Reranker-0.6B",
        collection_name: str = "beer_recipes",
    ):
        self.config = config

        # 1. Initialize LLM
        if config.llm_provider == LLMProvider.OPENROUTER:
            actual_model = model_name or config.openrouter_model
            logger.info(
                f"Initializing ChatOpenAI (OpenRouter) with model: {actual_model}"
            )
            self.llm = ChatOpenAI(
                model=actual_model,
                openai_api_key=config.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.3,
            )
        else:
            actual_model = model_name or "gemini-2.5-flash-lite"
            logger.info(
                f"Initializing ChatGoogleGenerativeAI with model: {actual_model}"
            )

            llm_kwargs = {
                "model": actual_model,
                "temperature": 0.3,
            }
            llm_kwargs["api_key"] = config.google_api_key
            self.llm = ChatGoogleGenerativeAI(**llm_kwargs)

        # 2. Initialize RAG Tool
        self.rag_tool = BeerRAGTool(
            config=config,
            model_name=embedding_model,
            collection_name=collection_name,
            rerank_model=rerank_model,
        )
        self.tools = [self.rag_tool]

        # 3. Setup Trimming Middleware
        @before_model
        def trim_history(state, config) -> dict | None:
            messages = state["messages"]

            if len(messages) <= 10:
                return None  # No changes needed

            trimmed = trim_messages(
                messages,
                strategy="last",
                token_counter=len,
                max_tokens=20,
                start_on="human",
                include_system=True,
            )

            return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]}

        # 4. Initialize Database Pool and Checkpointer
        self.psycopg_conn_str = config.connection_string.replace(
            "postgresql+psycopg://", "postgresql://"
        )

        # Run setup with autocommit=True to allow CREATE INDEX CONCURRENTLY
        try:
            with psycopg.connect(self.psycopg_conn_str, autocommit=True) as conn:
                setup_saver = PostgresSaver(conn)
                setup_saver.setup()
            logger.info("Checkpointer tables verified/created.")
        except Exception as e:
            logger.error(f"Failed to setup checkpointer tables: {e}")

        # Use a synchronous connection pool
        self.pool = ConnectionPool(
            self.psycopg_conn_str,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )

        # Use synchronous PostgresSaver
        self.saver = PostgresSaver(self.pool)

        # 5. Initialize and compile the Managed Agent once
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=(
                "You are RAG-on-Tap, an expert beer sommelier and master brewer. "
                "Your goal is to provide accurate, technical, and inspiring brewing advice. "
                "Use the provided 'search_beer_recipes' tool to find specific data whenever needed. "
                "When you provide information from a recipe, ALWAYS cite the Recipe Name and provide the Source URL. "
                "Be professional, encouraging, and accurate."
            ),
            middleware=[trim_history],
            checkpointer=self.saver,
        )
        logger.info(
            "ChatService initialized with persistent agent and sync connection pool."
        )

    def chat(self, user_input: str, session_id: str | None = None) -> str:
        """Sends a message to the agent and returns the response (Synchronously)."""
        if session_id is None:
            session_id = str(uuid.uuid4())

        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": session_id}},
            )
            return response["messages"][-1].content

        except Exception as e:
            logger.error(f"Error in ChatService ({session_id}): {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"

    def astream_chat(self, user_input: str, session_id: str):
        """Streams the agent response using a synchronous generator."""
        try:
            for msg, metadata in self.agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode="messages",
            ):
                logger.debug(
                    f"Stream yielded: type={msg.type} content_type={type(msg.content)}"
                )

                # Check for AI message type case-insensitively
                msg_type = str(msg.type).lower()
                if (msg_type == "ai" or msg_type == "aimessagechunk") and msg.content:
                    if isinstance(msg.content, str):
                        yield msg.content

        except Exception:
            logger.exception(f"Error in streaming ChatService ({session_id})")
            yield "\n[I encountered an error processing your request.]"
