import json
import os
import json
from fair_forge import Retriever
from fair_forge.schemas import Dataset

S3_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
S3_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET")
S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT")


class LocalRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        datasets=[]
        with open("dataset.json") as infile:
            for dataset in json.load(infile):
                datasets.append(Dataset.model_validate(dataset)) 
        return datasets
