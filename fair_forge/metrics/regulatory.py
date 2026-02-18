"""Regulatory metric for evaluating AI response compliance with regulations."""

from langchain_core.language_models.chat_models import BaseChatModel

from fair_forge.core import FairForge, Retriever
from fair_forge.llm import Judge, RegulatoryJudgeOutput
from fair_forge.llm.prompts import regulatory_reasoning_system_prompt
from fair_forge.schemas import Batch
from fair_forge.schemas.regulatory import RegulatoryMetric


class Regulatory(FairForge):
    """Metric for evaluating AI response compliance with regulations.

    Evaluates whether an assistant's responses comply with a given set of
    regulations, policies, or guidelines. This is useful for verifying
    compliance with banking/financial regulations, internal policies,
    industry guidelines, or any other rule-based requirements.

    Args:
        retriever: Retriever class for loading datasets
        model: LangChain BaseChatModel instance for evaluation
        regulations: List of regulations/rules to check compliance against
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        **kwargs: Additional arguments passed to FairForge base class
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        regulations: list[str],
        use_structured_output: bool = False,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.model = model
        self.regulations = regulations
        self.use_structured_output = use_structured_output
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause

        self.logger.info("--REGULATORY CONFIGURATION--")
        self.logger.info(f"Number of regulations: {len(self.regulations)}")
        self.logger.info(f"Structured output: {self.use_structured_output}")

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        """Process a batch of conversations for regulatory compliance.

        Args:
            session_id: Unique session identifier
            context: Context information for the conversation
            assistant_id: ID of the assistant being evaluated
            batch: List of Q&A interactions to evaluate
            language: Language of the conversation
        """
        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
            verbose=self.verbose,
        )

        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            # Format regulations as numbered list
            formatted_regulations = "\n".join(f"{i + 1}. {reg}" for i, reg in enumerate(self.regulations))

            data = {
                "regulations": formatted_regulations,
                "context": context,
                "assistant_answer": interaction.assistant,
            }

            reasoning, result = judge.check(
                regulatory_reasoning_system_prompt,
                interaction.query,
                data,
                output_schema=RegulatoryJudgeOutput,
            )

            if result is None:
                raise ValueError(f"[FAIR FORGE/REGULATORY] No valid response from judge for QA ID: {interaction.qa_id}")

            # Handle both Pydantic model and dict results
            if isinstance(result, dict):
                compliance_score = result["compliance_score"]
                insight = result["insight"]
                violated_rules = result.get("violated_rules", [])
                rule_assessments = result.get("rule_assessments", {})
            else:
                compliance_score = result.compliance_score
                insight = result.insight
                violated_rules = result.violated_rules
                rule_assessments = result.rule_assessments

            metric = RegulatoryMetric(
                session_id=session_id,
                assistant_id=assistant_id,
                qa_id=interaction.qa_id,
                compliance_score=compliance_score,
                compliance_insight=insight,
                compliance_thinkings=reasoning,
                violated_rules=violated_rules,
                rule_assessments=rule_assessments,
            )

            self.logger.debug(f"Compliance score: {metric.compliance_score}")
            self.logger.debug(f"Compliance insight: {metric.compliance_insight}")
            self.logger.debug(f"Violated rules: {metric.violated_rules}")
            self.metrics.append(metric)
