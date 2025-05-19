# Alquimia AI Fair Forge


<p align="center">
  <img src='https://www.alquimia.ai/logo-alquimia.svg' width=800>
</p>

**Alquimia AI Fair Forge** is a powerful performance-measurement component designed specifically for evaluating AI models and assistants. Provides clarity and insightful metrics, helping you understand and improve your AI applications.

---

## Getting Started

## How does it work?
Alquimia AI Fair Forge works under 5 main metrics:
- conversational
- humanity
- bias
- context

### Dataset

When you run Alquimia AI Fair Forge an specific dataset must be provided in order to properly work:
```json
[{
    "session_id": "123",
    "assistant_id": "456",
    "language": "english",
    "context": "You are a helpful assistant.",
    "conversation": [
        {
            "qa_id": "123",
            "query": "What is Alquimia AI?",
            "ground_truth_assistant": "Is an startup that its aim is to construct assistants",
            "assistant": "I'm so happy to answer your question. Alquimia AI Is an startup dedicated to construct assistants."
        }
    ]
}]

```
- Under `context` you specify where is the assistant being developed.
- `language` is to tell the humanity metric in which language should *ALWAYS* answer the assistant
- `assistant_id` tells Alquimia AI Fair Forge to which assistant should this conversation be made

Finally under the `conversation` we have each interaction of a user and assistant where:
- `user`: Refers to the human question
- `assistant` is the desired assistant answer used as ground truth to be evaluated later, in some cases we do not know the assistant answer so we can specify `observation` where we can tell Alquimia AI Fair Forge how it should answer (in an specific format for example)

### Set your custom retriever

In order to succesfully use your dataset with any of Fair Forge metrics you must set your custom retriever. 

```python
from fair_forge.schemas import Dataset
from fair_forge import Retriever

class CustomRetriever(Retriever):
    def load_dataset(self) -> list[Dataset]:
        datasets=[]
        with open("dataset.json") as infile:
            for dataset in json.load(infile):
                datasets.append(Dataset.model_validate(dataset)) 
        return datasets
```

## Use your desired metric
In order to properly use a metric you should instantiate and pass the custom retriever. 

```python
from getpass import getpass
from fair_forge.metrics import Context

judge_api_key = SecretStr(getpass("Please enter your Judge API key: "))

metrics = Context.run(
    CustomRetriever,
    judge_api_key=judge_api_key
)
```