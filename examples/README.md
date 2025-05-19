#### **Elastic mapping**

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
#### **Elastic mapping**

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

## Conversational Metric

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
