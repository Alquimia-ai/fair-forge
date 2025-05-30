import requests
import json
import requests
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
class AlquimiaRuntime():
    def __init__(self, url:str, token:str):
        self.token = token
        self.chat_endpoint = f"{url}/infer/chat"
        self.headers = {
            "Authorization": f"Bearer {self.token}"
        }
        self.stream_endpoint_template = f"{url}/stream/{{stream_id}}"
        
    def infer(self, assistant:str ,query:str,session_id:str,runtime:dict = {}):
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