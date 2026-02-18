"""Tests for LLM output schemas."""

import pytest
from pydantic import ValidationError

from fair_forge.llm.schemas import (
    BestOfJudgeOutput,
    ContextJudgeOutput,
    ConversationalJudgeOutput,
    RegulatoryJudgeOutput,
)


class TestContextJudgeOutput:
    """Tests for ContextJudgeOutput schema."""

    def test_valid_output(self):
        """Test valid context judge output."""
        output = ContextJudgeOutput(score=0.85, insight="Good context alignment")
        assert output.score == 0.85
        assert output.insight == "Good context alignment"

    def test_score_lower_bound(self):
        """Test score at lower bound."""
        output = ContextJudgeOutput(score=0.0, insight="Test")
        assert output.score == 0.0

    def test_score_upper_bound(self):
        """Test score at upper bound."""
        output = ContextJudgeOutput(score=1.0, insight="Test")
        assert output.score == 1.0

    def test_score_below_lower_bound(self):
        """Test score below lower bound raises error."""
        with pytest.raises(ValidationError):
            ContextJudgeOutput(score=-0.1, insight="Test")

    def test_score_above_upper_bound(self):
        """Test score above upper bound raises error."""
        with pytest.raises(ValidationError):
            ContextJudgeOutput(score=1.5, insight="Test")

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"score": 0.9, "insight": "Test insight"}
        output = ContextJudgeOutput.model_validate(data)
        assert output.score == 0.9
        assert output.insight == "Test insight"

    def test_missing_required_fields(self):
        """Test missing required fields raises error."""
        with pytest.raises(ValidationError):
            ContextJudgeOutput(score=0.5)

        with pytest.raises(ValidationError):
            ContextJudgeOutput(insight="Test")


class TestConversationalJudgeOutput:
    """Tests for ConversationalJudgeOutput schema."""

    def test_valid_output(self):
        """Test valid conversational judge output."""
        output = ConversationalJudgeOutput(
            memory=8.0,
            language=9.0,
            insight="Good conversation quality",
            quality_maxim=7.5,
            quantity_maxim=8.0,
            relation_maxim=9.0,
            manner_maxim=8.5,
            sensibleness=9.0,
        )
        assert output.memory == 8.0
        assert output.sensibleness == 9.0
        assert output.insight == "Good conversation quality"

    def test_all_scores_at_bounds(self):
        """Test all scores at bounds."""
        output = ConversationalJudgeOutput(
            memory=0.0,
            language=10.0,
            insight="Test",
            quality_maxim=0.0,
            quantity_maxim=10.0,
            relation_maxim=5.0,
            manner_maxim=5.0,
            sensibleness=10.0,
        )
        assert output.memory == 0.0
        assert output.language == 10.0

    def test_score_out_of_range(self):
        """Test score out of range raises error."""
        with pytest.raises(ValidationError):
            ConversationalJudgeOutput(
                memory=11.0,  # Invalid: > 10
                language=9.0,
                insight="test",
                quality_maxim=7.5,
                quantity_maxim=8.0,
                relation_maxim=9.0,
                manner_maxim=8.5,
                sensibleness=9.0,
            )

    def test_negative_score(self):
        """Test negative score raises error."""
        with pytest.raises(ValidationError):
            ConversationalJudgeOutput(
                memory=-1.0,  # Invalid: < 0
                language=9.0,
                insight="test",
                quality_maxim=7.5,
                quantity_maxim=8.0,
                relation_maxim=9.0,
                manner_maxim=8.5,
                sensibleness=9.0,
            )

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "memory": 7.0,
            "language": 8.0,
            "insight": "Test",
            "quality_maxim": 6.0,
            "quantity_maxim": 7.0,
            "relation_maxim": 8.0,
            "manner_maxim": 9.0,
            "sensibleness": 7.5,
        }
        output = ConversationalJudgeOutput.model_validate(data)
        assert output.memory == 7.0
        assert output.sensibleness == 7.5


class TestBestOfJudgeOutput:
    """Tests for BestOfJudgeOutput schema."""

    def test_valid_output(self):
        """Test valid best-of judge output."""
        output = BestOfJudgeOutput(
            winner="assistant_a",
            verdict="Better accuracy and completeness",
            confidence=0.9,
            reasoning={
                "assistant_a": {"strengths": ["accurate"], "weaknesses": ["verbose"]},
                "assistant_b": {"strengths": ["concise"], "weaknesses": ["incomplete"]},
            },
        )
        assert output.winner == "assistant_a"
        assert output.confidence == 0.9
        assert "assistant_a" in output.reasoning

    def test_tie_winner(self):
        """Test tie as winner."""
        output = BestOfJudgeOutput(
            winner="tie",
            verdict="Both responses equally good",
            confidence=0.5,
            reasoning={},
        )
        assert output.winner == "tie"

    def test_confidence_bounds(self):
        """Test confidence at bounds."""
        output_low = BestOfJudgeOutput(winner="a", verdict="Test", confidence=0.0, reasoning={})
        assert output_low.confidence == 0.0

        output_high = BestOfJudgeOutput(winner="a", verdict="Test", confidence=1.0, reasoning={})
        assert output_high.confidence == 1.0

    def test_confidence_out_of_range(self):
        """Test confidence out of range raises error."""
        with pytest.raises(ValidationError):
            BestOfJudgeOutput(winner="a", verdict="Test", confidence=1.5, reasoning={})

        with pytest.raises(ValidationError):
            BestOfJudgeOutput(winner="a", verdict="Test", confidence=-0.1, reasoning={})

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "winner": "model_x",
            "verdict": "Superior performance",
            "confidence": 0.85,
            "reasoning": {"model_x": {"strengths": ["fast"]}},
        }
        output = BestOfJudgeOutput.model_validate(data)
        assert output.winner == "model_x"
        assert output.confidence == 0.85


class TestRegulatoryJudgeOutput:
    """Tests for RegulatoryJudgeOutput schema."""

    def test_valid_output(self):
        """Test valid regulatory judge output."""
        output = RegulatoryJudgeOutput(
            compliance_score=0.95,
            insight="Fully compliant with all regulations",
            violated_rules=[],
            rule_assessments={
                "1": {"compliant": True, "reason": "Verified identity"},
                "2": {"compliant": True, "reason": "No data sharing"},
            },
        )
        assert output.compliance_score == 0.95
        assert output.insight == "Fully compliant with all regulations"
        assert output.violated_rules == []
        assert len(output.rule_assessments) == 2

    def test_with_violations(self):
        """Test regulatory output with violations."""
        output = RegulatoryJudgeOutput(
            compliance_score=0.5,
            insight="Some rules violated",
            violated_rules=["1", "3"],
            rule_assessments={
                "1": {"compliant": False, "reason": "No identity check"},
                "2": {"compliant": True, "reason": "OK"},
                "3": {"compliant": False, "reason": "Shared data without consent"},
            },
        )
        assert output.compliance_score == 0.5
        assert output.violated_rules == ["1", "3"]
        assert output.rule_assessments["1"]["compliant"] is False

    def test_compliance_score_lower_bound(self):
        """Test compliance score at lower bound."""
        output = RegulatoryJudgeOutput(
            compliance_score=0.0,
            insight="Total non-compliance",
            violated_rules=["all"],
            rule_assessments={},
        )
        assert output.compliance_score == 0.0

    def test_compliance_score_upper_bound(self):
        """Test compliance score at upper bound."""
        output = RegulatoryJudgeOutput(
            compliance_score=1.0,
            insight="Perfect compliance",
            violated_rules=[],
            rule_assessments={},
        )
        assert output.compliance_score == 1.0

    def test_compliance_score_below_lower_bound(self):
        """Test compliance score below lower bound raises error."""
        with pytest.raises(ValidationError):
            RegulatoryJudgeOutput(
                compliance_score=-0.1,
                insight="Test",
                violated_rules=[],
                rule_assessments={},
            )

    def test_compliance_score_above_upper_bound(self):
        """Test compliance score above upper bound raises error."""
        with pytest.raises(ValidationError):
            RegulatoryJudgeOutput(
                compliance_score=1.5,
                insight="Test",
                violated_rules=[],
                rule_assessments={},
            )

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "compliance_score": 0.75,
            "insight": "Mostly compliant",
            "violated_rules": ["rule_x"],
            "rule_assessments": {"rule_x": {"compliant": False, "reason": "Failed check"}},
        }
        output = RegulatoryJudgeOutput.model_validate(data)
        assert output.compliance_score == 0.75
        assert output.violated_rules == ["rule_x"]

    def test_missing_required_fields(self):
        """Test missing required fields raises error."""
        with pytest.raises(ValidationError):
            RegulatoryJudgeOutput(compliance_score=0.5)

        with pytest.raises(ValidationError):
            RegulatoryJudgeOutput(
                compliance_score=0.5,
                insight="Test",
                # Missing violated_rules and rule_assessments
            )
