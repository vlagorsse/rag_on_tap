from unittest.mock import MagicMock, patch

import pytest
from langchain.tools import BaseTool
from services.chat_service import ChatService
from services.config_service import ConfigService, LLMProvider


class TestChatService:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock(spec=ConfigService)
        config.google_api_key = "fake_key"
        config.connection_string = "postgresql+psycopg://user:pass@host/db"
        config.llm_provider = LLMProvider.GOOGLE
        return config

    @patch("services.chat_service.psycopg")
    @patch("services.chat_service.PostgresChatMessageHistory")
    @patch("services.chat_service.ConversationBufferWindowMemory")
    @patch("services.chat_service.ChatGoogleGenerativeAI")
    @patch("services.chat_service.BeerRAGTool", spec=BaseTool)
    @patch("services.chat_service.AgentExecutor")
    def test_chat_invocation(
        self,
        mock_executor_class,
        mock_tool_class,
        mock_llm_class,
        mock_memory_class,
        mock_pg_history,
        mock_psycopg,
        mock_config,
    ):
        """Test that ChatService invokes the agent correctly."""
        # Setup mocks
        mock_executor = mock_executor_class.return_value
        mock_executor.invoke.return_value = {"output": "Mocked AI Response"}

        # Mock connection context manager
        mock_conn = MagicMock()
        mock_psycopg.connect.return_value.__enter__.return_value = mock_conn

        service = ChatService(config=mock_config)
        response = service.chat("Hello")

        assert response == "Mocked AI Response"
        mock_executor.invoke.assert_called_once()

        # Verify memory was passed
        args, kwargs = mock_executor_class.call_args
        assert "memory" in kwargs

    @patch("services.chat_service.psycopg")
    @patch("services.chat_service.PostgresChatMessageHistory")
    @patch("services.chat_service.ChatGoogleGenerativeAI")
    @patch("services.chat_service.BeerRAGTool", spec=BaseTool)
    def test_initialization(
        self,
        mock_tool_class,
        mock_llm_class,
        mock_pg_history,
        mock_psycopg,
        mock_config,
    ):
        """Test that services are initialized with correct parameters."""
        service = ChatService(
            config=mock_config, model_name="test-model", embedding_model="test-embed"
        )

        # Check LLM init
        mock_llm_class.assert_called_once()
        _, llm_kwargs = mock_llm_class.call_args
        assert llm_kwargs["model"] == "test-model"

        # Check Tool init
        mock_tool_class.assert_called_once()
        _, tool_kwargs = mock_tool_class.call_args
        assert tool_kwargs["model_name"] == "test-embed"
