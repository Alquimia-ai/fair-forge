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

    def batch(
        self,
        session_id: str,
        context: str,
        agent_id: str,
        batch: list[Batch],
        language: str | None = "english",
    ):
        # BestOf processes in _process() instead of batch()
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

    def _build_conversation_batch(self, assistant_id: str) -> str:
        """Build a formatted string from conversations for a specific assistant."""
        assistant_conversations = []
        for dataset in self.dataset:
            if dataset.assistant_id == assistant_id:
                assistant_conversations.extend(dataset.conversation)

        template = Template(bestOf_contestant_format)
        return template.render(conversations=assistant_conversations)

    def _process(self) -> list:
        """Override base processing for tournament-style comparison."""
        self.logger.info("[BestOf] Aggregating datasets by query for best-of comparisons")

        judge = Judge(
            model=self.model,
            use_structured_output=self.use_structured_output,
            bos_json_clause=self.bos_json_clause,
            eos_json_clause=self.eos_json_clause,
        )

        assistant_ids = sorted({dataset.assistant_id for dataset in self.dataset})
        current_contestants = assistant_ids.copy()

        round_number = 1
        contests: list[BestOfContest] = []

        while len(current_contestants) > 1:
            self.logger.info(f"[BestOf] Round {round_number}: {len(current_contestants)} contestants remaining")
            pairs = []
            round_winners = []
            num_contestants = len(current_contestants)

            if num_contestants % 2 == 1:
                bye_contestant = current_contestants[-1]
                self.logger.info(f"[BestOf] Round {round_number}: {bye_contestant} receives a bye")
                round_winners.append(bye_contestant)
                iterable_limit = num_contestants - 1
            else:
                iterable_limit = num_contestants

            for i in range(0, iterable_limit, 2):
                pairs.append((current_contestants[i], current_contestants[i + 1]))
            self.logger.info(f"[BestOf] Round {round_number}: Comparing {len(pairs)} pairs")

            for pair in pairs:
                left_id, right_id = pair

                left_contestant = self._build_conversation_batch(left_id)
                right_contestant = self._build_conversation_batch(right_id)

                self.logger.debug(f"Round {round_number}: Comparing {left_id} and {right_id} contestants")
                reasoning, result = judge.check(
                    bestOf_judge_prompt,
                    self.criteria,
                    {
                        "left_contestant": left_id,
                        "right_contestant": right_id,
                        "left_contestant_conv": left_contestant,
                        "right_contestant_conv": right_contestant,
                    },
                    output_schema=BestOfJudgeOutput,
                )

                if result is None:
                    raise ValueError(f"[FAIR FORGE/BESTOF] No valid response from judge for {left_id} vs {right_id}")

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

                if winner == left_id:
                    round_winners.append(left_id)
                elif winner == right_id:
                    round_winners.append(right_id)
                elif isinstance(winner, str) and winner.lower() == "tie":
                    round_winners.append(left_id)
                    round_winners.append(right_id)
                else:
                    raise ValueError(f"[FAIR FORGE/BESTOF] Winner name doesn't match {winner} {left_id} {right_id}")

                self.logger.debug(f"Round {round_number}: Winner is {winner}")
                contests.append(
                    BestOfContest(
                        round=round_number,
                        left_id=left_id,
                        right_id=right_id,
                        winner_id=winner if winner in (left_id, right_id) else "tie",
                        confidence=confidence,
                        verdict=verdict,
                        reasoning=result_reasoning,
                        thinkings=reasoning,
                    )
                )

            self.logger.info(f"[BestOf] Round {round_number} winners: {round_winners}")

            current_contestants = round_winners
            round_number += 1

        if len(current_contestants) == 1:
            final_winner = current_contestants[0]
            self.logger.info(f"[BestOf] Tournament complete! Final winner: {final_winner}")
            tournament_metric = BestOfMetric(
                session_id="bestof",
                qa_id="bestof_tournament",
                assistant_id=final_winner,
                bestof_winner_id=final_winner,
                bestof_contests=contests,
            )
            self.metrics.append(tournament_metric)
        else:
            self.logger.warning("[BestOf] Tournament ended in a tie or unresolved state")
            tournament_metric = BestOfMetric(
                session_id="bestof",
                qa_id="bestof_tournament",
                assistant_id="tie",
                bestof_winner_id="tie",
                bestof_contests=contests,
            )
            self.metrics.append(tournament_metric)

        return self.metrics
