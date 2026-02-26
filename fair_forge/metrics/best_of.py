"""BestOf metric for tournament-style evaluation of AI responses."""

from jinja2 import Template
from langchain_core.language_models.chat_models import BaseChatModel

from fair_forge.core import FairForge, Retriever
from fair_forge.llm import BestOfJudgeOutput, Judge
from fair_forge.llm.prompts import bestOf_contestant_format, bestOf_judge_prompt
from fair_forge.schemas import Batch
from fair_forge.schemas.best_of import BestOfContest, BestOfMetric


class BestOf(FairForge):
    """Tournament-style metric for comparing multiple AI assistant responses.

    Args:
        retriever: Retriever class for loading datasets
        model: LangChain BaseChatModel instance for evaluation
        use_structured_output: If True, use LangChain's with_structured_output()
        bos_json_clause: Opening marker for JSON blocks
        eos_json_clause: Closing marker for JSON blocks
        criteria: Label describing the evaluation criteria
        **kwargs: Additional arguments passed to FairForge base class
    """

    def __init__(
        self,
        retriever: type[Retriever],
        model: BaseChatModel,
        use_structured_output: bool = False,
        bos_json_clause: str = "```json",
        eos_json_clause: str = "```",
        criteria: str = "BestOf",
        **kwargs,
    ):
        super().__init__(retriever, **kwargs)
        self.model = model
        self.use_structured_output = use_structured_output
        self.bos_json_clause = bos_json_clause
        self.eos_json_clause = eos_json_clause
        self.criteria = criteria
        self._session_kings = {}
        self._judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
        )

    def _build_single_contestant(self, batch: list[Batch]) -> str:
        """Build a formatted string from a specific batch of conversations."""
        template = Template(bestOf_contestant_format)
        return template.render(conversations=batch)

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        if not batch:
            return

        block_key = tuple(interaction.qa_id for interaction in batch)
        incoming_text = self._build_single_contestant(batch)

        if block_key not in self._session_kings:
            self.logger.debug(f"[BestOf] Setting {assistant_id} as initial King for block {block_key}")
            self._session_kings[block_key] = (assistant_id, incoming_text, [])
            return

        king_id, king_text, matches = self._session_kings[block_key]

        self.logger.info(f"[BestOf] Challenging King {king_id} vs {assistant_id} for block {block_key}")

        reasoning, result = self._judge.check(
            bestOf_judge_prompt,
            self.criteria,
            {
                "left_contestant": king_id,
                "right_contestant": assistant_id,
                "left_contestant_conv": king_text,
                "right_contestant_conv": incoming_text,
            },
            output_schema=BestOfJudgeOutput,
        )

        if result is None:
            raise ValueError(f"[FAIR FORGE/BESTOF] No valid response from judge for {king_id} vs {assistant_id}")

        if self.use_structured_output:
            winner = result.winner
            confidence = result.confidence
            verdict = result.verdict
            result_reasoning = result.reasoning
        else:
            winner = result.get("winner", "")
            confidence = result.get("confidence")
            verdict = result.get("verdict")
            result_reasoning = result.get("reasoning")

        match_record = BestOfContest(
            round=len(matches) + 1,
            left_id=king_id,
            right_id=assistant_id,
            winner_id=winner if winner in (king_id, assistant_id) else "tie",
            confidence=confidence,
            verdict=verdict,
            reasoning=result_reasoning,
            thinkings=reasoning,
        )

        if winner == assistant_id:
            self.logger.info(f"[BestOf] New King! {assistant_id} dethrones {king_id}")
            self._session_kings[block_key] = (assistant_id, incoming_text, [*matches, match_record])
        elif winner == king_id:
            self.logger.info(f"[BestOf] {king_id} defends the throne against {assistant_id}")
            self._session_kings[block_key] = (king_id, king_text, [*matches, match_record])
        else:
            self.logger.warning(f"[BestOf] Match between {king_id} and {assistant_id} ended in a tie. King stays.")
            self._session_kings[block_key] = (king_id, king_text, [*matches, match_record])

    def on_process_complete(self):
        """Evaluate the final kings and emit the metrics."""
        self.logger.info(f"[BestOf] Stream complete. Emitting {len(self._session_kings)} tournament results.")

        for block_key, (king_id, _, matches) in self._session_kings.items():
            qa_id_repr = block_key[0] if len(block_key) == 1 else f"batch_len_{len(block_key)}"
            if matches:
                final_match = matches[-1]
                winner_id = final_match.winner_id
            else:
                winner_id = king_id

            metric = BestOfMetric(
                session_id="bestof",
                qa_id=qa_id_repr,
                assistant_id=winner_id,
                bestof_winner_id=winner_id,
                bestof_contests=matches,
            )
            self.metrics.append(metric)
