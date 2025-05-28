import json
import os
import json
import io
from fair_forge.schemas import Dataset
from fair_forge import Retriever
import lakefs
from lakefs.client import Client

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lakefs_host= os.environ.get("LAKEFS_HOST")
        self.lakefs_username= os.environ.get("LAKEFS_USERNAME")
        self.lakefs_password= os.environ.get("LAKEFS_PASSWORD")
        self.lakefs_repository= os.environ.get("repository")
        self.lakefs_branch= os.environ.get("branch")
        self.lakefs_dataset= os.environ.get("dataset")

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
            "LAKEFS_HOST": self.lakefs_host,
            "LAKEFS_USERNAME": self.lakefs_username,
            "LAKEFS_PASSWORD": self.lakefs_password,
            "LAKEFS_REPOSITORY": self.lakefs_repository,
            "LAKEFS_BRANCH": self.lakefs_branch,
            "LAKEFS_DATASET": self.lakefs_dataset
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]


        if missing_vars:
            raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Initialize LakeFS client
        client = Client(
            host=self.lakefs_host,
            username=self.lakefs_username,
            password=self.lakefs_password
        )
        
        # Get repository and reference
        repo = lakefs.Repository(repository_id=self.lakefs_repository, client=client)
        ref = repo.ref(self.lakefs_branch)
        
        # Get the dataset object
        obj = ref.object(path=self.lakefs_dataset)
        
        datasets = []
        # Read and parse the dataset
        with obj.reader(mode="rb") as raw_reader:
            # Wrap the binary stream in a text wrapper
            text_reader = io.TextIOWrapper(raw_reader, encoding="utf-8")
            for dataset in json.load(text_reader):
                datasets.append(Dataset.model_validate(dataset))
        
        return datasets
        
