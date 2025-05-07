import requests
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from .dataset import load_raw_dataset,Conversation,Batch
import json
import re
import requests
import time
import uuid
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Judge():
    def __init__(self):
        self.chat_history = []
        self.reasoning_model = ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
    def reason(self, system_prompt: str, query: str, data: dict):
        self.chat_history.append(("human", query))
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            *self.chat_history
        ])
        chain = prompt | self.reasoning_model
        response = chain.invoke(data)
        reasoning = response.content
    
        think_match = re.search(r"<think>(.*?)</think>", reasoning, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', reasoning, re.DOTALL)
        json_str = json_match.group(1).strip() if json_match else ""
        json_data = None
        if json_str:
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                json_data = None
                logging.error(f"JSON decoding error: {e}")
        else:
            logging.error(reasoning)
        
        
        return think_content, json_data
        

class AlquimiaRuntime():
    def __init__(self, url:str, token:str):
        self.token = token
        self.chat_endpoint = f"{url}/infer/chat"
        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }
        self.stream_endpoint_template = f"{url}/stream/{{stream_id}}"
        self.reasoning_model = ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    def _infer(self, assistant:str ,query:str,session_id:str,runtime:dict = {}):
        payload = {
            "query": query,
            "session_id": session_id,
            **runtime
        }
        response = requests.post(f"{self.chat_endpoint}/{assistant}", json=payload, headers=self.headers)
        data = response.json()['data']
        stream_id = data.get("stream_id")
        if not stream_id:
            logging.error("Error: No se recibió stream_id")
            return
        final_answer = ""
        is_complete = False
        while not is_complete:
            stream_url = self.stream_endpoint_template.format(stream_id=stream_id)
            stream_resp = requests.get(stream_url, headers=self.headers, stream=True)
            for line in stream_resp.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data:"):
                        json_line = decoded_line[len("data:"):].strip()
                        try:
                            json_data = json.loads(json_line)
                        except Exception:
                            continue
                        if json_data.get("is_complete", False) or json_data.get("is_completed", False):
                            is_complete = True
                            break
            if not is_complete:
                time.sleep(1)
        if 'error_code' in json_data:
            raise Exception(json_data['error_detail'])
            
        return json_data

    def infer_dataset(self,dataset_path:str):
        dataset = load_raw_dataset(dataset_path)
        logging.info("[RUNTIME] Starting to infer dataset")
        conversations = []
        for thread in dataset:
            session_id = str(uuid.uuid4())
            
            if thread.args.get('extra_data'):
                logging.info(f"[RUNTIME] It must use extra_data: {thread.args['extra_data']}")
        
            if thread.args.get('force_profile'):
                logging.info(f"[RUNTIME] It must use force_profile: {thread.args['force_profile']}")
            
            logging.info(f"[RUNTIME/{session_id}] Starting Conversation")
            conv_batches = []
            conversation = Conversation(context=thread.context,assistant_id=thread.assistant,preferred_language= thread.preferred_language,session_id=session_id)
            for batch in thread.conversation:
                query = batch.user
                inference = self._infer(thread.assistant, query, session_id, thread.args)
                leviathan = {k: v for k, v in inference.items() if k != 'answer' and k!='is_complete'}
                logging.info(f"[RUNTIME/{session_id}] Human: {query}\n\nThinkings:\n{leviathan}\n\nAssistant: {inference['answer']}")
                processed_batch = Batch(ground_truth_assistant=batch.assistant,
                                        observation=batch.observation,
                                        assistant=inference['answer'],
                                        question=query,
                                        ground_truth_leviathan=batch.leviathan,
                                        leviathan = leviathan,
                                        qa_id = str(uuid.uuid4()))
                conv_batches.append(processed_batch)
            conversation.conversation = conv_batches
            conversations.append(conversation.model_dump())
        return conversations