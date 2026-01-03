"""Tests for Chain of Thought module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import SecretStr
from fair_forge.llm.cot import CoT, ChainOfThought


class TestChainOfThought:
    """Test suite for ChainOfThought model."""

    def test_chain_of_thought_model(self):
        """Test ChainOfThought Pydantic model."""
        cot = ChainOfThought(thought="This is thinking", answer="This is answer")
        assert cot.thought == "This is thinking"
        assert cot.answer == "This is answer"

    def test_chain_of_thought_empty_thought(self):
        """Test ChainOfThought with empty thought."""
        cot = ChainOfThought(thought="", answer="Answer only")
        assert cot.thought == ""
        assert cot.answer == "Answer only"


class TestCoT:
    """Test suite for CoT class."""

    @patch('fair_forge.llm.cot.ChatOpenAI')
    def test_initialization_default(self, mock_chat_openai):
        """Test CoT initialization with default parameters."""
        cot = CoT()
        assert cot.bos_think_token == "<think>"
        assert cot.eos_think_token == "</think>"
        assert cot.chat_history == []
        mock_chat_openai.assert_called_once()

    @patch('fair_forge.llm.cot.ChatOpenAI')
    def test_initialization_custom_tokens(self, mock_chat_openai):
        """Test CoT initialization with custom thinking tokens."""
        cot = CoT(
            bos_think_token="<reasoning>",
            eos_think_token="</reasoning>"
        )
        assert cot.bos_think_token == "<reasoning>"
        assert cot.eos_think_token == "</reasoning>"

    @patch('fair_forge.llm.cot.ChatOpenAI')
    def test_initialization_custom_params(self, mock_chat_openai):
        """Test CoT initialization with custom parameters."""
        cot = CoT(
            base_url="https://custom-api.com",
            api_key=SecretStr("test-key"),
            model="custom-model",
            temperature=0.5
        )
        mock_chat_openai.assert_called_once_with(
            base_url="https://custom-api.com",
            api_key=SecretStr("test-key"),
            model="custom-model",
            temperature=0.5,
        )

    @patch('fair_forge.llm.cot.ChatOpenAI')
    @patch('fair_forge.llm.cot.ChatPromptTemplate')
    def test_reason_with_thinking(self, mock_template, mock_chat_openai):
        """Test reason method extracts thinking and answer."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        # Mock response with thinking tokens
        mock_response = MagicMock()
        mock_response.content = "<think>This is my reasoning process</think>The final answer is 42."
        mock_chain.invoke.return_value = mock_response

        cot = CoT()
        result = cot.reason("You are a helpful assistant", "What is the answer?")

        assert isinstance(result, ChainOfThought)
        assert result.thought == "This is my reasoning process"
        assert result.answer == "The final answer is 42."

    @patch('fair_forge.llm.cot.ChatOpenAI')
    @patch('fair_forge.llm.cot.ChatPromptTemplate')
    def test_reason_without_thinking(self, mock_template, mock_chat_openai):
        """Test reason method when no thinking tokens present."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        # Mock response without thinking tokens
        mock_response = MagicMock()
        mock_response.content = "Direct answer without thinking"
        mock_chain.invoke.return_value = mock_response

        cot = CoT()
        result = cot.reason("You are a helpful assistant", "What is the answer?")

        assert isinstance(result, ChainOfThought)
        assert result.thought == ""
        assert result.answer == "Direct answer without thinking"

    @patch('fair_forge.llm.cot.ChatOpenAI')
    @patch('fair_forge.llm.cot.ChatPromptTemplate')
    def test_reason_adds_to_chat_history(self, mock_template, mock_chat_openai):
        """Test that reason adds queries to chat history."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_chain.invoke.return_value = mock_response

        cot = CoT()
        assert len(cot.chat_history) == 0

        cot.reason("System", "Query 1")
        assert len(cot.chat_history) == 1
        assert cot.chat_history[0] == ("human", "Query 1")

        cot.reason("System", "Query 2")
        assert len(cot.chat_history) == 2
        assert cot.chat_history[1] == ("human", "Query 2")

    @patch('fair_forge.llm.cot.ChatOpenAI')
    @patch('fair_forge.llm.cot.ChatPromptTemplate')
    def test_reason_with_kwargs(self, mock_template, mock_chat_openai):
        """Test reason method passes kwargs to invoke."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_chain.invoke.return_value = mock_response

        cot = CoT()
        cot.reason("System", "Query", custom_param="value")

        mock_chain.invoke.assert_called_once_with({'custom_param': 'value'})

    @patch('fair_forge.llm.cot.ChatOpenAI')
    @patch('fair_forge.llm.cot.ChatPromptTemplate')
    def test_reason_custom_think_tokens(self, mock_template, mock_chat_openai):
        """Test reason with custom thinking tokens."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        # Use default tokens for this test
        mock_response = MagicMock()
        mock_response.content = "<think>My reasoning</think>The answer"
        mock_chain.invoke.return_value = mock_response

        cot = CoT(bos_think_token="<think>", eos_think_token="</think>")
        result = cot.reason("System", "Query")

        assert result.thought == "My reasoning"
        assert result.answer == "The answer"
