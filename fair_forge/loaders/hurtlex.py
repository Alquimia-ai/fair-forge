"""HurtLex toxicity dataset loader."""
import pkg_resources
import pandas as pd
from fair_forge.core.loader import ToxicityLoader
from fair_forge.schemas.toxicity import ToxicityDataset


class HurtlexLoader(ToxicityLoader):
    """
    Loads HurtLex multilingual toxicity lexicon.

    HurtLex is a lexicon of offensive, aggressive, and hateful words
    in multiple languages.
    """

    def load(self, language: str) -> list[ToxicityDataset]:
        """
        Load HurtLex toxicity dataset for a specific language.

        Args:
            language: Language code (e.g., "english", "spanish")

        Returns:
            List of ToxicityDataset entries
        """
        df = pd.read_csv(
            pkg_resources.resource_filename(
                "fair_forge",
                f"artifacts/toxicity/hurtlex_{language}.tsv"
            ),
            sep="\t",
            header=0,
        )
        return [
            ToxicityDataset(word=row["lemma"], category=row["category"])
            for _, row in df.iterrows()
        ]
