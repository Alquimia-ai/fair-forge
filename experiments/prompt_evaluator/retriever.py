"""JSON retriever for PromptEvaluator experiment datasets."""

import json
from pathlib import Path

from fair_forge.core.retriever import Retriever
from fair_forge.schemas.common import Dataset


class JsonRetriever(Retriever):
    def __init__(self, path: Path | str, **kwargs):
        super().__init__(**kwargs)
        self._path = Path(path)

    def load_dataset(self) -> list[Dataset]:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [Dataset.model_validate(entry) for entry in data]
        return [Dataset.model_validate(data)]
