"""Common retrievers for Fair Forge examples."""

import json
from pathlib import Path

from fair_forge import Retriever
from fair_forge.schemas import Dataset


class LocalRetriever(Retriever):
    """Retriever that loads datasets from a local JSON file."""

    def __init__(self, dataset_path: str | None = None, **kwargs):
        """Initialize the LocalRetriever.

        Args:
            dataset_path: Path to the dataset JSON file. If not provided,
                defaults to examples/data/dataset.json.
            **kwargs: Additional arguments passed to base Retriever.
        """
        super().__init__(**kwargs)
        if dataset_path:
            self.dataset_path = Path(dataset_path)
        else:
            examples_dir = Path(__file__).parent.parent
            self.dataset_path = examples_dir / "data" / "dataset.json"

    def load_dataset(self) -> list[Dataset]:
        """Load datasets from the JSON file.

        Returns:
            List of Dataset objects parsed from the JSON file.
        """
        datasets = []
        with open(self.dataset_path) as infile:
            for dataset in json.load(infile):
                datasets.append(Dataset.model_validate(dataset))
        return datasets
