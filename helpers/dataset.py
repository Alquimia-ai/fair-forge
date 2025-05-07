import json
from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel, parse_obj_as
import os
import boto3

S3_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
S3_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
S3_BUCKET_NAME = os.environ.get("AWS_S3_BUCKET")
S3_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT")

class Batch(BaseModel):
    qa_id: str
    ground_truth_assistant: Optional[str] ## What is supposed to be the correct assistant answer
    observation: Optional[str] # In some cases such as the tools output as we do not know which is going to be the answer we add an observation
    assistant: str ## What it actually responded
    question: str
    leviathan: dict
    ground_truth_leviathan: dict

class Conversation(BaseModel):
    context: str
    preferred_language: str
    session_id: str
    assistant_id: str
    conversation: List[Batch] = []


### Dataset retrieved from S3
class RawConversation(BaseModel):
    user: str
    assistant: Optional[str] = None
    observation: Optional[str] = None
    leviathan: dict

class Thread(BaseModel):
    context: str
    preferred_language: str
    args: dict
    assistant: str
    conversation: List[RawConversation]

def load_raw_dataset(dataset_name: str) -> Tuple[List[Thread], Dict]:
    """
    Loads JSON files from S3 in the given prefix (dataset_name).
    Returns a tuple:
      ( [Thread objects from main dataset],
        {extra_data_dict if extra_data.json exists, else {} } )
    """
    if (S3_SECRET_ACCESS_KEY is None 
            or S3_ACCESS_KEY_ID is None 
            or S3_BUCKET_NAME is None 
            or S3_ENDPOINT is None):
        raise TypeError("One or more required environment variables are not set.")

    session = boto3.session.Session()
    s3_client = session.client(
        's3',
        region_name='nyc3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY
    )

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=f"{dataset_name}")

    main_data = []

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj.get('Key', '')
            if not key.endswith('.json'):
                continue

            data = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            content = data['Body'].read().decode("utf-8")
            main_data.extend(json.loads(content))

    threads = parse_obj_as(List[Thread], main_data)

    return threads

def load_dataset() -> List[Conversation]:
    with open("dataset.json", "r") as infile:
        data = json.load(infile)
        conversations = []
        for conversation in data:
            conversations.append(Conversation.parse_obj(conversation))
        return conversations