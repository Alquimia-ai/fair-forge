from collections import defaultdict
from typing import Type, Optional, Any
import pandas as pd
from fair_forge.core import FairForge, Retriever
from fair_forge.schemas import Batch, HumanityMetric
import re
import math
import numpy as np
from scipy.stats import spearmanr
import pkg_resources


# TODO:
# - Implement Emotion matching
# - Implement Language Style Matching, LSM
# - Implement Agreeableness
# - Implement Empathy, Empathic Concern

class Humanity(FairForge):
    def __init__(self, retriever: Type[Retriever], **kwargs):
        super().__init__(retriever, **kwargs)
        self.emotion_columns = [
            "Anger",
            "Anticipation",
            "Disgust",
            "Fear",
            "Joy",
            "Sadness",
            "Surprise",
            "Trust",
        ]

    def _load_emotion_lexicon(
        self,
        path: Optional[str] = pkg_resources.resource_filename(
            "fair_forge", "artifacts/lexicons/nrc_emotion.csv"
        ),
        separator: Optional[str] = ";",
        language: Optional[str] = "english",
    ):
        nrc = pd.read_csv(str(path), sep=separator, encoding="utf-8")
        lexicon = {}
        for index, row in nrc.iterrows():
            word = str(row[language]).lower()
            emotions = [e for e in self.emotion_columns if row[e] == 1]
            lexicon[word] = emotions
        return lexicon

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def _get_emotion_distribution(self, text: str, lexicon, emotion_list):
        counts = defaultdict(
            int
        )  ## Creates a  dictionary that if no index found returns 0
        total = 0
        for word in self._tokenize(text):
            if word in lexicon:
                for emotion in lexicon[word]:
                    counts[emotion] += 1
                    total += 1

        if total == 0:
            return {emotion: 0 for emotion in emotion_list}

        return {
            emotion: counts[emotion] / total for emotion in emotion_list
        }  # frequency / total

    def _emotional_entropy(self, distribution):
        entropy = 0
        for p in distribution.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def batch(
        self,
        session_id: str,
        context: str,
        assistant_id: str,
        batch: list[Batch],
        language: Optional[str] = "english",
        path: Optional[str] = pkg_resources.resource_filename(
            "fair_forge", "artifacts/lexicons/nrc_emotion.csv"
        ),
    ):
        lexicon = self._load_emotion_lexicon(path=path, language=language)
        for interaction in batch:
            self.logger.debug(f"QA ID: {interaction.qa_id}")

            assistant_distribution = self._get_emotion_distribution(
                interaction.assistant, lexicon, self.emotion_columns
            )
            
            generated_vec = [assistant_distribution[e] for e in self.emotion_columns]
            self.logger.debug(f"Assistant distribution: {assistant_distribution}")
            self.logger.debug(f"Generated vector: {generated_vec}")
            ## Execute emotional entropy
            ent = self._emotional_entropy(assistant_distribution)
            spearman_val: float = 0.0
            if interaction.ground_truth_assistant is not None:
                ground_truth_assistant_distribution = self._get_emotion_distribution(
                    interaction.ground_truth_assistant, lexicon, self.emotion_columns
                )
                ## Spearman correlation between ground truth and real assistant answer
                expected_vec = [
                    ground_truth_assistant_distribution[e] for e in self.emotion_columns
                ]

                ## If the standard deviation is 0, the correlation is not defined
                if np.std(generated_vec) == 0 or np.std(expected_vec) == 0:
                    spearman_val = 0
                else:
                    result: Any = spearmanr(expected_vec, generated_vec)
                    spearman_val = result.correlation
            metric = HumanityMetric(
                    session_id=session_id,
                    qa_id=interaction.qa_id,
                    assistant_id=assistant_id,
                    humanity_assistant_emotional_entropy=ent,
                    humanity_ground_truth_spearman=round(spearman_val, 3),
                    **{
                        f"humanity_assistant_{key.lower()}": assistant_distribution[key]
                        for key in self.emotion_columns
                    },
                )
            self.logger.debug(f"Spearman value: {metric.humanity_ground_truth_spearman}")
            self.logger.debug(f"Emotional entropy: {metric.humanity_assistant_emotional_entropy}")
            for key in self.emotion_columns:
                emotion_key = f"humanity_assistant_{key.lower()}"
                self.logger.debug(f"{key}: {getattr(metric, emotion_key)}")
            self.metrics.append(metric)
