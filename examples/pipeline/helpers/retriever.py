import json
import os
import json
import io
from fair_forge.schemas import Dataset
from fair_forge import Retriever
import lakefs
from lakefs.client import Client

LAKEFS_HOST=os.environ.get("LAKEFS_HOST")
LAKEFS_USERNAME= os.environ.get("LAKEFS_USERNAME")
LAKEFS_PASSWORD= os.environ.get("LAKEFS_PASSWORD")
LAKEFS_REPOSITORY= os.environ.get("LAKEFS_REPOSITORY")
LAKEFS_BRANCH= os.environ.get("LAKEFS_BRANCH")
LAKEFS_DATASET=os.environ.get("LAKEFS_DATASET")

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
    
class LakeFSRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        """
        Load dataset from LakeFS storage.
        
        This method connects to LakeFS using environment variables for configuration,
        retrieves the dataset file, and converts it into a list of Dataset objects.
        
        Returns:
            list[Dataset]: A list of Dataset objects containing the retrieved data.
            
        Raises:
            Exception: If required environment variables are missing or if there are issues
                     connecting to LakeFS or reading the dataset.
        """
        # Validate required environment variables
        required_vars = {
            "LAKEFS_HOST": LAKEFS_HOST,
            "LAKEFS_USERNAME": LAKEFS_USERNAME,
            "LAKEFS_PASSWORD": LAKEFS_PASSWORD,
            "LAKEFS_REPOSITORY": LAKEFS_REPOSITORY,
            "LAKEFS_BRANCH": LAKEFS_BRANCH,
            "LAKEFS_DATASET": LAKEFS_DATASET
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize LakeFS client
        client = Client(
            host=LAKEFS_HOST,
            username=LAKEFS_USERNAME,
            password=LAKEFS_PASSWORD
        )
        
        # Get repository and reference
        repo = lakefs.Repository(repository_id=LAKEFS_REPOSITORY, client=client)
        ref = repo.ref(LAKEFS_BRANCH)
        
        # Get the dataset object
        obj = ref.object(path=LAKEFS_DATASET)
        
        datasets = []
        # Read and parse the dataset
        with obj.reader(mode="rb") as raw_reader:
            # Wrap the binary stream in a text wrapper
            text_reader = io.TextIOWrapper(raw_reader, encoding="utf-8")
            for dataset in json.load(text_reader):
                datasets.append(Dataset.model_validate(dataset))
        
        return datasets
        
