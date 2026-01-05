"""Tests for Judge module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pydantic import SecretStr
from fair_forge.llm.judge import Judge
from fair_forge.llm.cot import ChainOfThought


class TestJudge:
    """Test suite for Judge class."""

    @patch('fair_forge.llm.judge.ChatOpenAI')
    def test_initialization_without_cot(self, mock_chat_openai):
        """Test Judge initialization without Chain of Thought."""
        judge = Judge()
        assert judge.cot is None
        assert judge.chat_history == []
        assert judge.bos_json_clause == "```json"
        assert judge.eos_json_clause == "```"
        mock_chat_openai.assert_called_once()

    @patch('fair_forge.llm.judge.CoT')
    def test_initialization_with_cot(self, mock_cot):
        """Test Judge initialization with Chain of Thought tokens."""
        judge = Judge(bos_think_token="<think>", eos_think_token="</think>")
        assert judge.cot is not None
        mock_cot.assert_called_once_with(
            bos_think_token="<think>",
            eos_think_token="</think>",
            base_url="https://api.groq.com/openai",
            api_key=SecretStr(""),
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
        )

    @patch('fair_forge.llm.judge.ChatOpenAI')
    def test_initialization_custom_json_clauses(self, mock_chat_openai):
        """Test Judge initialization with custom JSON clauses."""
        judge = Judge(bos_json_clause="<json>", eos_json_clause="</json>")
        assert judge.bos_json_clause == "<json>"
        assert judge.eos_json_clause == "</json>"

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_direct_inference(self, mock_template, mock_chat_openai):
        """Test direct_inference method."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = "This is the response"
        mock_chain.invoke.return_value = mock_response

        judge = Judge()
        result = judge.direct_inference("System prompt", "User query")

        assert isinstance(result, ChainOfThought)
        assert result.thought == ""
        assert result.answer == "This is the response"
        assert ("human", "User query") in judge.chat_history

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_check_without_cot_valid_json(self, mock_template, mock_chat_openai):
        """Test check method without CoT returning valid JSON."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = 'Here is the result:\n```json\n{"score": 0.85, "valid": true}\n```'
        mock_chain.invoke.return_value = mock_response

        judge = Judge()
        thought, json_data = judge.check("System prompt", "Query", {"key": "value"})

        assert thought == ""
        assert json_data == {"score": 0.85, "valid": True}

    @patch('fair_forge.llm.judge.CoT')
    def test_check_with_cot_valid_json(self, mock_cot):
        """Test check method with CoT returning valid JSON."""
        mock_cot_instance = MagicMock()
        mock_cot.return_value = mock_cot_instance
        mock_cot_instance.reason.return_value = ChainOfThought(
            thought="I need to analyze this",
            answer='```json\n{"result": "success"}\n```'
        )

        judge = Judge(bos_think_token="<think>", eos_think_token="</think>")
        thought, json_data = judge.check("System", "Query", {})

        assert thought == "I need to analyze this"
        assert json_data == {"result": "success"}

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_check_no_json_found(self, mock_template, mock_chat_openai, caplog):
        """Test check method when no JSON is found."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = "Response without JSON"
        mock_chain.invoke.return_value = mock_response

        judge = Judge()
        thought, json_data = judge.check("System", "Query", {})

        assert thought == ""
        assert json_data is None

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_check_invalid_json(self, mock_template, mock_chat_openai, caplog):
        """Test check method with invalid JSON."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = '```json\n{invalid json}\n```'
        mock_chain.invoke.return_value = mock_response

        judge = Judge()
        thought, json_data = judge.check("System", "Query", {})

        assert thought == ""
        assert json_data is None

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_check_custom_json_clauses(self, mock_template, mock_chat_openai):
        """Test check method with custom JSON clauses."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = 'Result: <json>{"value": 42}</json>'
        mock_chain.invoke.return_value = mock_response

        judge = Judge(bos_json_clause="<json>", eos_json_clause="</json>")
        thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"value": 42}

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_direct_inference_with_kwargs(self, mock_template, mock_chat_openai):
        """Test direct_inference passes kwargs to invoke."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = "Response"
        mock_chain.invoke.return_value = mock_response

        judge = Judge()
        judge.direct_inference("System", "Query", custom_key="custom_value")

        mock_chain.invoke.assert_called_once_with({'custom_key': 'custom_value'})

    @patch('fair_forge.llm.judge.ChatOpenAI')
    def test_chat_history_accumulates(self, mock_chat_openai):
        """Test that chat history accumulates across calls."""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        with patch('fair_forge.llm.judge.ChatPromptTemplate') as mock_template:
            mock_chain = MagicMock()
            mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)
            mock_response = MagicMock()
            mock_response.content = "Response"
            mock_chain.invoke.return_value = mock_response

            judge = Judge()
            assert len(judge.chat_history) == 0

            judge.direct_inference("System", "Query 1")
            assert len(judge.chat_history) == 1

            judge.direct_inference("System", "Query 2")
            assert len(judge.chat_history) == 2

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_check_json_with_whitespace(self, mock_template, mock_chat_openai):
        """Test check handles JSON with extra whitespace."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = '```json   \n  {"key": "value"}  \n  ```'
        mock_chain.invoke.return_value = mock_response

        judge = Judge()
        thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"key": "value"}

    @patch('fair_forge.llm.judge.ChatOpenAI')
    @patch('fair_forge.llm.judge.ChatPromptTemplate')
    def test_check_nested_json(self, mock_template, mock_chat_openai):
        """Test check handles nested JSON."""
        mock_model = MagicMock()
        mock_chain = MagicMock()
        mock_chat_openai.return_value = mock_model
        mock_template.from_messages.return_value.__or__ = Mock(return_value=mock_chain)

        mock_response = MagicMock()
        mock_response.content = '```json\n{"outer": {"inner": [1, 2, 3]}}\n```'
        mock_chain.invoke.return_value = mock_response

        judge = Judge()
        thought, json_data = judge.check("System", "Query", {})

        assert json_data == {"outer": {"inner": [1, 2, 3]}}
