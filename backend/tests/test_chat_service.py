from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

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

    @patch("services.chat_service.ConnectionPool")
    @patch("services.chat_service.PostgresSaver")
    @patch("services.chat_service.ChatGoogleGenerativeAI")
    @patch("services.chat_service.BeerRAGTool")
    @patch("services.chat_service.create_agent")
    @patch("services.chat_service.psycopg")
    def test_chat_invocation(
        self,
        mock_psycopg,
        mock_create_agent,
        mock_tool_class,
        mock_llm_class,
        mock_saver_class,
        mock_pool_class,
        mock_config,
    ):
        """Test that ChatService invokes the agent correctly."""
        # Setup mocks
        mock_tool = mock_tool_class.return_value
        mock_tool.name = "search_beer_recipes"

        mock_agent = mock_create_agent.return_value
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Mocked AI Response")]
        }

        service = ChatService(config=mock_config)
        response = service.chat("Hello")

        assert response == "Mocked AI Response"
        mock_agent.invoke.assert_called_once()

        # Verify thread_id was passed in config
        args, kwargs = mock_agent.invoke.call_args
        assert "config" in kwargs
        assert "thread_id" in kwargs["config"]["configurable"]

    @patch("services.chat_service.ConnectionPool")
    @patch("services.chat_service.PostgresSaver")
    @patch("services.chat_service.ChatGoogleGenerativeAI")
    @patch("services.chat_service.BeerRAGTool")
    @patch("services.chat_service.create_agent")
    @patch("services.chat_service.psycopg")
    def test_initialization(
        self,
        mock_psycopg,
        mock_create_agent,
        mock_tool_class,
        mock_llm_class,
        mock_saver_class,
        mock_pool_class,
        mock_config,
    ):
        """Test that services are initialized with correct parameters."""
        mock_tool = mock_tool_class.return_value
        mock_tool.name = "search_beer_recipes"

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

        # Check pool init
        mock_pool_class.assert_called_once()
