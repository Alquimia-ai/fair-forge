# Alquimia AI Fair Forge


<p align="center">
  <img src='https://www.alquimia.ai/logo-alquimia.svg' width=800>
</p>

**Alquimia AI Fair Forge** is a powerful performance-measurement component designed specifically for evaluating AI models and assistants. Provides clarity and insightful metrics, helping you understand and improve your AI applications.

---

## Getting Started

### Requirements
To execute **Alquimia AI Fair Forge**, ensure that a pipeline server is installed through either:
- **Openshift AI**
- **Tekton Pipelines**

### Configuration

1. Open the `logos.pipeline` file.
2. Under each notebook in `node_properties` you are going to have the required env variables

### Env variables
```shell
ELASTIC_URL=https://host:port
ELASTIC_AUTH_USER=elastic
ELASTIC_AUTH_PASSWORD=YOUR_PASSWORD
ALQUIMIA_RUNTIME_URL=https://runtime.alquimiaai.hostmydemo.online
ALQUIMIA_TOKEN=YOUR-SECRET
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_S3_BUCKET=alquimia-ai-fair-forge
AWS_S3_ENDPOINT=http://minio_url:port
GROQ_API_KEY=your-api-key
GUARDIAN_URL=https://runtime.alquimiaai.hostmydemo.online
GUARDIAN_MODEL_NAME=granite2b
GUARDIAN_API_KEY=YOUR_API_KEY
GUARDIAN_MODEL=ibm-granite/granite-guardian-3.1-2b
```

## Deploying the Pipeline

After configuring your environment correctly, export your pipeline as a YAML file. You can then execute it via:
- OpenShift AI Panel
- Tekton Pipelines

At runtime, you will be prompted to specify these parameters:
| Parameter             | Description                                                         | Example            |
|-----------------------|---------------------------------------------------------------------|--------------------|
| `dataset`        | Name of the dataset folder stored in S3                                    | `customer-queries` |
| `ngram_w2v`           | N-gram size for constructing word embeddings                        | `1`                |

## How does it work?
Alquimia AI Fair Forge works under 5 main metrics:
- conversational
- humanity
- bias
- agentic
- context

### Dataset

When you run Alquimia AI Fair Forge an specific dataset must be provided in order to properly work:
```json
[
  {
    "context": "The ASB Banking Assistant has been developed with the goal of providing highly accurate, coherent, and context-aware responses based on information available from various sources, such as uploaded documents, historical records, and direct user inquiries. Its processing capabilities enable real-time data analysis, identification of relevant patterns, and the generation of well-founded responses, ensuring that the information provided is reliable and relevant for financial decision-making.  The Assistant Responses Showcase functionality is designed to demonstrate the potential of artificial intelligence in the banking sector, ensuring that responses are not only technically correct but also comprehensible and adapted to the specific context of each query. Through advanced natural language processing (NLP) techniques and machine learning, the assistant can interpret different types of requests, apply information filters based on predefined criteria, and generate reports or visualizations in a structured and easy-to-interpret format.  Furthermore, ASB maintains a strong focus on user experience (UX) and visual interface (UI), delivering information in a clear, organized, and accessible manner. It dynamically adapts to conversations, maintaining contextual coherence throughout interactions and allowing users to explore data in greater depth. The transparency in information presentation, combined with real-time update capabilities, ensures that responses reflect the most recent data and provide valuable insights for strategic decision-making.",
    "preferred_language": "english",
    "args": {
      "force_profile": {
        "_collection_id": "asb-empresas"
      }
    },
    "assistant": "asb-financial",
    "conversation": [
      {
        "user": "Give me summary statistics of loan amounts.",
        "leviathan": {
          "tool_summary": {
            "name": "query_dataframe",
            "parameters": {
              "sql_query": "SELECT MIN(LOAN_AMOUNT) AS min_amount, MAX(LOAN_AMOUNT) AS max_amount, AVG(LOAN_AMOUNT) AS avg_amount, COUNT(LOAN_ID) AS num_loans FROM loans;"
            }
          }
        },
        "assistant": "The lowest recorded loan amount is $200,000, while the highest reaches $700,000. On average, the loans granted amount to $422,500. In total, 20 loans have been issued, adding up to a total of $8,450,000."
      },
      {
        "user": "Can you provide delinquency and risk analysis?",
        "leviathan": {
          "tool_summary": {
            "name": "query_dataframe",
            "parameters": {
              "sql_query": "SELECT CREDIT_STATUS, COUNT(*) AS total_loans, SUM(LOAN_AMOUNT) AS total_amount FROM loans GROUP BY CREDIT_STATUS;"
            }
          }
        },
        "assistant": "There are currently 5 loans with an 'Excellent' credit status, amounting to $2,650,000 in total. Additionally, 7 loans fall under the 'Good' credit status, summing up to $3,150,000. Lastly, we have 8 loans classified as 'Regular', with a combined total of $2,650,000."
      }
    ]
  }
]
```
- Under `context` you specify where is the assistant being developed.
- `preferred_language` is to tell the humanity metric in which language should *ALWAYS* answer the assistant
- `assistant` tells Alquimia AI Fair Forge to which assistant should this conversation be made
- `args` is used for alquimia-runtime to specify some overrides

Finally under the `conversation` we have each interaction of a user and assistant where:
- `user`: Refers to the human question
- `leviathan` aims at specifying which are the desired outputs of each intermediate model such as tool_analyst,intent, sentiment or any other
- `assistant` is the desired assistant answer used as ground truth to be evaluated later, in some cases we do not know the assistant answer so we can specify `observation` where we can tell Alquimia AI Fair Forge how it should answer (in an specific format for example)

#### **Elastic mapping**
In elastic each interaction between a human and an assistant is mapped as a single doc, so for example:
```json
{
        "user": "Give me summary statistics of loan amounts.",
        "leviathan": {
          "tool_summary": {
            "name": "query_dataframe",
            "parameters": {
              "sql_query": "SELECT MIN(LOAN_AMOUNT) AS min_amount, MAX(LOAN_AMOUNT) AS max_amount, AVG(LOAN_AMOUNT) AS avg_amount, COUNT(LOAN_ID) AS num_loans FROM loans;"
            }
          }
        },
        "assistant": "The lowest recorded loan amount is $200,000, while the highest reaches $700,000. On average, the loans granted amount to $422,500. In total, 20 loans have been issued, adding up to a total of $8,450,000."
}
```

Is going to be a single doc where:
```json
{
  "session_id": {"type": "keyword"},
  "observation": {"type": "text"},
  "assistant": {"type": "text"},
  "ground_truth_assistant": {"type": "text"},
  "question": {"type": "text"},
  "qa_id": {"type": "keyword"},
  "assistant_id": {"type":"keyword"}
}
```
- session_id: Identifies to which conversation thread belongs these interaction
- qa_id: Identifies this interaction
- question: The human query
- ground_truth_asisstant: What is was expected to be answered (can be null if only observation was specified)
- observation: The observation added in the dataset (can be null if only the assistant was specified)
- assistant: This is the real answer inferred from the assistant

### Context Metric

This metric aims at using the `context` from each conversation to check if the assistant answer adjusts to the given or provided context. To properly calculate this metric we focus on using an `LLM As a Judge` that receives context, human question, assistant answer and ground truth assistant answer (The one expected ) or observation. Finally this llm assigns an score of how much is the real assistant attached to the context based on the ground truth/observation, alongside this score an `insight` is also provided.

As we use CoT models (specifically deepseek-r1) the full reasoning process is provided so the thinking process is also saved. Finally we have this mapping in elastic:

```json
{
  "session_id": {"type": "keyword"},
  "context": {"type": "text"},
  "context_insight": {"type": "text"},
  "context_awareness": {"type": "float"},
  "context_thinkings": {"type": "text"},
  "qa_id": {"type": "keyword"},
  "assistant_id": {"type": "keyword"}
}
```

The context metric also runs a word embedding process that for each session_id saves calculate the corresponding embedding. Then through a TSNE process we obtain the 3-dimensionalities of each word, in elastic this properties are saved also as docs like this:
```json
{
  "word": {"type": "text"},
  "x": {"type": "float"},
  "y": {"type": "float"},
  "z": {"type": "float"},
  "session_id": {"type": "keyword"},
  "assistant_id": {"type": "keyword"}
}
```
## Conversational Metric

This metric aims at anaylizing any data related to conversation. In order to properly handle this type of metric several criterias were set up:
- Using an `llm as a judge` we check the memory (how well the model perform in terms of remembering). It goes from 0 to 10 as an scoring
- `language score` assigns a value from 0 to 10 of how well did the assistant address the language barrier, taking into account the the preferred language
- In social science generally and linguistics specifically, the cooperative principle describes how people achieve effective conversational communication in common social situations—that is, how listeners and speakers act cooperatively and mutually accept one another to be understood in a particular way. The concept of the cooperative principle was introduced by the linguist Paul Grice in his pragmatic theory. Grice researched the ways in which people derive meaning from language. Alquimia AI Fair Forge retrieve 4 maxims as [Grice's Maxims](https://www.sas.upenn.edu/~haroldfs/dravling/grice.html) specified
- `sensibleness` is a metric that takes inspiration from google's [Sensibleness and specificity Average](https://arxiv.org/abs/2001.09977)
- Finally inside the metric `conversational_thinkings` and `conversational_insight` the whole thinking process of the CoT model is retrieved and also the insight.

#### Elastic mapping
```json

{
  "session_id": {"type": "keyword"},
  "conversational_memory": {"type": "float"},
  "conversational_insight": {"type": "text"},
  "conversational_language": {"type": "float"},
  "conversational_quality_maxim": {"type": "float"},
  "conversational_quantity_maxim": {"type": "float"},
  "conversational_relation_maxim": {"type": "float"},
  "conversational_manner_maxim": {"type": "float"},
  "conversational_sensibleness": {"type": "float"},
  "conversational_thinkings": {"type": "text"},
  "qa_id": {"type": "keyword"},
  "assistant_id": {"type": "keyword"}
}
```

## Humanity Metric
For this type of metric we are trying to measure how 'human' does it feel to talk with the assistant. In order to properly handle this several metric are taken into account

- `Emotional entropy`: Following [Psychological Metrics for Dialog evaluations](https://arxiv.org/pdf/2305.14757) we used the [NRC Emotion Lexicon Dataset](https://nrc-publications.canada.ca/eng/view/object/?id=0b6a5b58-a656-49d3-ab3e-252050a7a88c) that gives for each word in a given vocabulary a set of emotions, which are motivated by [Plutchik Wheel of emotions](https://www.6seconds.org/2025/02/06/plutchik-wheel-emotions/). Then given the answer from the assistant we find a emotional distribution that basically says the probability of each of the 8 emotions to appear in the assistant real answer. Finally the [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of that emotion distribution is calculated.
- `Ground truth spearman`: Using the same dataset from before we calculate the emotional distribution from the assistant answer and the emotional distribution from the ground truth assistant answer, we then use spearman's correlation to study how well does the emotion of the assistant aligns to the expected emotion

#### Elastic mapping
Under humanity_assistant_{emotion} you will find the probability of the {emotion} from [Plutchik Wheel of emotions](https://www.6seconds.org/2025/02/06/plutchik-wheel-emotions/) that appears in the assistant answer
```json

{
  "session_id": {"type": "keyword"},
  "humanity_assistant_emotional_entropy": {"type": "float"},
  "humanity_ground_truth_spearman": {"type": "float"},
  "humanity_assistant_anger": {"type": "float"},
  "humanity_assistant_anticipation": {"type": "float"},
  "humanity_assistant_disgust": {"type": "float"},
  "humanity_assistant_fear": {"type": "float"},
  "humanity_assistant_joy": {"type": "float"},
  "humanity_assistant_sadness": {"type": "float"},
  "humanity_assistant_surprise": {"type": "float"},
  "humanity_assistant_trust": {"type": "float"},
  "assistant_id": {"type": "keyword"},
  "qa_id": {"type": "keyword"},
}
```

## Bias Metric
Following towards a more robust assistant's architecture, we made an integration with [Granite Guardian](https://huggingface.co/ibm-granite/granite-guardian-3.1-2b) to check that all human and assistant interactions do not follow any risk set in [AI ATLAS](https://www.ibm.com/docs/en/watsonx/saas?topic=ai-risk-atlas). 

Granite Guardian is useful for risk detection use-cases which are applicable across a wide-range of enterprise applications 
- Detecting harm-related risks within prompt text or model response (as guardrails). These present two fundamentally different use cases as the former assesses user supplied text while the latter evaluates model generated text.
- RAG (retrieval-augmented generation) use-case where the guardian model assesses three key issues: context relevance (whether the retrieved context is relevant to the query), groundedness (whether the response is accurate and faithful to the provided context), and answer relevance (whether the response directly addresses the user's query).
- Function calling risk detection within agentic workflows, where Granite Guardian evaluates intermediate steps for syntactic and semantic hallucinations. This includes assessing the validity of function calls and detecting fabricated information, particularly during query translation.

#### Elastic mapping
```json
{
  "session_id": {"type": "keyword"},
  "bias_guard_is_risk": {"type": "boolean"},
  "bias_guard_type": {"type": "text"},
  "bias_guard_probability": {"type": "float"},
  "assistant_id": {"type": "keyword"},
  "qa_id": {"type": "keyword"},
}
```

## Agentic Metric
This is the simplest of the metric, yet the most useful. Basically takes the `ground_truth_leviathan` from the dataset and compares to the `leviathan` proper answer from the assistant.

#### Elastic mapping
```json
{
  "agentic_leviathan_type": {"type": "text"},
  "agentic_leviathan_ground_truth": {"type": "object"},
  "agentic_leviathan": {"type": "object"},
  "assistant_id": {"type": "keyword"},
  "qa_id": {"type": "keyword"},
  "assistant_id": {"type":"keyword"}
}
```
