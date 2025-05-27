import json
import os
import json
from fair_forge.schemas import Dataset
from fair_forge import Retriever

S3_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
S3_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET")
S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT")

class LocalRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        datasets = []
        # Get the absolute path to examples/dataset.json
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
        examples_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach examples
        dataset_path = os.path.join(examples_dir, "dataset.json")
        
        with open(dataset_path) as infile:
            for dataset in json.load(infile):
                datasets.append(Dataset.model_validate(dataset)) 
        return datasets
