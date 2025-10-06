from fair_forge import FairForge, Retriever
from typing import Type, Optional
from fair_forge.schemas import Batch, BestOfMetric, BestOfContest
from fair_forge.helpers.judge import Judge
from fair_forge.prompts import bestOf_contestant_format, bestOf_judge_prompt
from pydantic import SecretStr
from jinja2 import Template

class BestOf(FairForge):
    def __init__(self, retriever: Type[Retriever],
        judge_base_url: str = "https://api.groq.com/openai/v1",
        judge_api_key: SecretStr = SecretStr(""),
        judge_model: str = "deepseek-r1-distill-llama-70b",
        judge_temperature: float = 0,
        judge_bos_json_clause: str = "```json",
        judge_eos_json_clause: str = "```",
        judge_bos_think_token: Optional[str] = None,
        judge_eos_think_token: Optional[str] = None,
        criteria:str = "BestOf",
        **kwargs,):
        """Initialize a BestOf evaluator.

        Parameters:
            retriever (Type[Retriever]): Concrete `Retriever` class used by
                `FairForge` to obtain context/evidence.
            judge_base_url (str): Base URL for a judge-compatible OpenAI API
                endpoint.
            judge_api_key (SecretStr): API key used to authenticate judge API
                requests.
            judge_model (str): Model identifier to use for the judge.
            judge_temperature (float): Sampling temperature for judge
                generations.
            judge_bos_json_clause (str): Opening fence/marker that precedes
                JSON output returned by the judge.
            judge_eos_json_clause (str): Closing fence/marker that follows
                JSON output returned by the judge.
            judge_bos_think_token (Optional[str]): Optional token that marks
                the beginning of a chain-of-thought segment in judge output.
            judge_eos_think_token (Optional[str]): Optional token that marks
                the end of a chain-of-thought segment in judge output.
            criteria (str): Label describing the evaluation criteria for this
                arena run.
            **kwargs: Additional keyword arguments forwarded to `FairForge`.
        """
        super().__init__(retriever, **kwargs)
        self.judge_url = judge_base_url
        self.judge_api_key = judge_api_key
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature
        self.judge_bos_think_token = judge_bos_think_token
        self.judge_eos_think_token = judge_eos_think_token
        self.judge_bos_json_clause = judge_bos_json_clause
        self.judge_eos_json_clause = judge_eos_json_clause
        self.criteria = criteria

    def batch(
        self,
        session_id: str,
        context: str,
        agent_id: str,
        batch: list[Batch],
        language: Optional[str] = "english",
    ):  
        judge = Judge(
                bos_think_token=self.judge_bos_think_token,
                eos_think_token=self.judge_eos_think_token,
                base_url=self.judge_url,
                api_key=self.judge_api_key,
                model=self.judge_model,
                temperature=self.judge_temperature,
                bos_json_clause=self.judge_bos_json_clause,
                eos_json_clause=self.judge_eos_json_clause,
        )
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")
            pass


    def _build_conversation_batch(self, assistant_id: str) -> str:
        """
        Build a formatted string from conversations for a specific assistant using Jinja2 template.
        
        Args:
            assistant_id (str): The assistant ID to filter conversations for
            
        Returns:
            str: Formatted conversation string
        """
        assistant_conversations = []
        for dataset in self.dataset:
            if dataset.assistant_id == assistant_id:
                assistant_conversations.extend(dataset.conversation)
        
        template = Template(bestOf_contestant_format)
        formatted_output = template.render(conversations=assistant_conversations)
        
        return formatted_output

    def _process(self) -> list:
        """
        Override base processing to aggregate all datasets by query and
        perform pairwise best-of comparisons across assistants for the same input.

        Returns:
            list: BestOf comparison records appended to self.metrics
        """
        self.logger.info("[BestOf] Aggregating datasets by query for best-of comparisons")

        ## Initialize judge
        judge = Judge(
            bos_think_token=self.judge_bos_think_token,
            eos_think_token=self.judge_eos_think_token,
            base_url=self.judge_url,
            api_key=self.judge_api_key,
            model=self.judge_model,
            temperature=self.judge_temperature,
            bos_json_clause=self.judge_bos_json_clause,
            eos_json_clause=self.judge_eos_json_clause,
        )

        # no symmetric duplicates
        assistant_ids = sorted({dataset.assistant_id for dataset in self.dataset})
        current_contestants = assistant_ids.copy()
        
        # tournament
        round_number = 1
        records = []
        contests: list[BestOfContest] = []
        
        while len(current_contestants) > 1:
            self.logger.info(f"[BestOf] Round {round_number}: {len(current_contestants)} contestants remaining")
            # Pair contestants in bracket order to avoid duplicate/self matchups
            pairs = []
            round_winners = []
            num_contestants = len(current_contestants)
            # If odd number, give the last contestant a bye to next round
            if num_contestants % 2 == 1:
                bye_contestant = current_contestants[-1]
                self.logger.info(f"[BestOf] Round {round_number}: {bye_contestant} receives a bye")
                round_winners.append(bye_contestant)
                iterable_limit = num_contestants - 1
            else:
                iterable_limit = num_contestants
            # Create adjacent pairs: (0,1), (2,3), ...
            for i in range(0, iterable_limit, 2):
                pairs.append((current_contestants[i], current_contestants[i + 1]))
            self.logger.info(f"[BestOf] Round {round_number}: Comparing {len(pairs)} pairs")
            
            for pair in pairs:
                left_id, right_id = pair
                
                left_contestant = self._build_conversation_batch(left_id)
                right_contestant = self._build_conversation_batch(right_id)
                
                self.logger.debug(f"Round {round_number}: Comparing {left_id} and {right_id} contestants")
                thinking, json = judge.check(bestOf_judge_prompt, self.criteria, {"left_contestant": left_id, "right_contestant": right_id, "left_contestant_conv": left_contestant, "right_contestant_conv": right_contestant})
                if json is None:
                    raise ValueError(
                        f"[FAIR FORGE/CONTEXT] No JSON found {self.judge_bos_json_clause} {self.judge_eos_json_clause} "
                    )
                
                # Track the winner
                winner = json.get("winner", "")
                
                if winner == left_id:
                    round_winners.append(left_id)
                elif winner == right_id:
                    round_winners.append(right_id)
                elif isinstance(winner, str) and winner.lower() == "tie":
                    # In case of tie, advance both contestants to the next round
                    round_winners.append(left_id)
                    round_winners.append(right_id)
                else:
                    raise ValueError(f"[FAIR FORGE/CONTEXT] Winner name doesn't match {winner} {left_id} {right_id}")
                
                self.logger.debug(f"Round {round_number}: Winner is {winner}")
                contests.append(
                    BestOfContest(
                        round=round_number,
                        left_id=left_id,
                        right_id=right_id,
                        winner_id=winner if winner in (left_id, right_id) else "tie",
                        confidence=json.get("confidence"),
                        verdict=json.get("verdict"),
                        reasoning=json.get("reasoning"),
                        thinkings=thinking,
                    )
                )
                records.append({"thinking":thinking, **json})
            
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
            # If more than one contestant remains, consider it a tie outcome for the tournament
            self.logger.warning("[BestOf] Tournament ended in a tie or unresolved state; marking final winner as 'tie'")
            tournament_metric = BestOfMetric(
                session_id="bestof",
                qa_id="bestof_tournament",
                assistant_id="tie",
                bestof_winner_id="tie",
                bestof_contests=contests,
            )
            self.metrics.append(tournament_metric)
        
        return self.metrics
