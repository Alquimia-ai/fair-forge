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
  "mappings": {
    "properties": {
      "session_id": {"type": "keyword"},
      "assistant_id": {"type": "keyword"},
      "confidence_intervals": {
        "type": "nested",
        "properties": {
          "protected_attribute": {"type": "keyword"},
          "lower_bound": {"type": "float"},
          "upper_bound": {"type": "float"},
          "probability": {"type": "float"},
          "samples": {"type": "integer"},
          "k_success": {"type": "integer"},
          "alpha": {"type": "float"},
          "confidence_level": {"type": "float"}
        }
      },
      "guardian_interactions": {
        "type": "nested",
        "properties": {
          "protected_attribute": {"type": "keyword"},
          "qa_id": {"type": "keyword"},
          "is_biased": {"type": "boolean"},
          "certainty": {"type": "float"}
        }
      },
      "cluster_profiling": {
        "type": "object",
        "properties": {
          "cluster_id": {"type": "keyword"},
          "toxicity_score": {"type": "float"}
        }
      },
      "assistant_space": {
        "properties": {
          "cluster_labels": {"type": "keyword"},
          "embeddings": {"type": "float"},
          "latent_space": {"type": "float"}
        }
      }
    }
  }
}
```

The mapping includes:
- `session_id` and `assistant_id`: Identifiers for the session and assistant
- `confidence_intervals`: Nested object containing statistical confidence intervals for protected attributes
- `guardian_interactions`: Nested object tracking bias detection results from the guardian
- `cluster_profiling`: Object containing cluster analysis results
- `assistant_space`: Object containing embedding and latent space information
