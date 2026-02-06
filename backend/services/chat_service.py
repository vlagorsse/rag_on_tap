import contextlib
import logging
import psycopg

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres import PostgresChatMessageHistory

from services.config_service import ConfigService
from services.rag_tool import BeerRAGTool

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        config: ConfigService,
        model_name: str = "gemini-2.5-flash-lite",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        rerank_model: str = "Qwen/Qwen3-Reranker-0.6B",
        collection_name: str = "beer_recipes",
        max_token_limit: int = 1000,
    ):
        self.config = config

        # 1. Initialize LLM
        logger.info(f"Initializing ChatGoogleGenerativeAI with model: {model_name}")

        llm_kwargs = {
            "model": model_name,
            "temperature": 0.3,
        }
        # Check if key is set and not empty
        api_key = config.google_api_key
        if api_key and api_key.strip():
            llm_kwargs["api_key"] = api_key
        else:
            logger.warning("GOOGLE_API_KEY is not set or empty in ConfigService, relying on environment variables.")

        self.llm = ChatGoogleGenerativeAI(**llm_kwargs)
        # 2. Initialize RAG Tool
        self.rag_tool = BeerRAGTool(
            config=config,
            model_name=embedding_model,
            collection_name=collection_name,
            rerank_model=rerank_model,
        )
        self.tools = [self.rag_tool]

        # 3. Setup Prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are RAG-on-Tap, an expert beer sommelier and master brewer. "
                        "Your goal is to provide accurate, technical, and inspiring brewing advice "
                        "Use the provided 'search_beer_recipes' tool to find specific data whenever needed. "
                        "When you provide information from a recipe, ALWAYS cite the Recipe Name and provide the Source URL. "
                        "Be professional, encouraging, and accurate."
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # 4. Shared Agent (stateless)
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.max_token_limit = max_token_limit

        # 5. Database Setup for Memory
        self.connection_string = config.connection_string
        # PostgresChatMessageHistory.create_tables needs a connection, not a string
        # We also strip the sqlalchemy driver prefix for psycopg
        self.psycopg_conn_str = self.connection_string.replace("postgresql+psycopg://", "postgresql://")
        try:
            with psycopg.connect(self.psycopg_conn_str) as conn:
                PostgresChatMessageHistory.create_tables(conn, "chat_history")
            logger.info("Chat history tables verified/created.")
        except Exception as e:
            logger.error(f"Failed to initialize chat history tables: {e}")

    @contextlib.contextmanager
    def _session_executor(self, session_id: str):
        """Context manager that provides a transient AgentExecutor with a scoped DB connection."""
        conn = None
        try:
            conn = psycopg.connect(self.psycopg_conn_str)
            chat_history = PostgresChatMessageHistory(
                "chat_history",
                session_id,
                sync_connection=conn,
            )

            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                max_token_limit=self.max_token_limit,
                memory_key="chat_history",
                chat_memory=chat_history,
                return_messages=True
            )

            executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,
                handle_parsing_errors=True,
                memory=memory,
            )
            yield executor
        finally:
            if conn:
                conn.close()

    def chat(self, user_input: str, session_id: str = "default") -> str:
        """Sends a message to the agent and returns the response."""
        try:
            with self._session_executor(session_id) as executor:
                response = executor.invoke({"input": user_input})
                return response["output"]

        except Exception as e:
            logger.error(f"Error in ChatService ({session_id}): {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"

    async def astream_chat(self, user_input: str, session_id: str = "default"):
        """Asynchronously streams the agent response."""
        try:
            with self._session_executor(session_id) as executor:
                async for chunk in executor.astream({"input": user_input}):
                    # LangChain agent streaming returns different types of chunks
                    if "actions" in chunk:
                        # Intermediate tool calls
                        for action in chunk["actions"]:
                            logger.info(f"Agent ({session_id}) calling tool: {action.tool}")
                    elif "steps" in chunk:
                        # Results from tools
                        pass
                    elif "output" in chunk:
                        # Final output part
                        yield chunk["output"]

        except Exception as e:
            logger.error(f"Error in streaming ChatService ({session_id}): {e}")
            yield f"\n[Error: {str(e)}]"
