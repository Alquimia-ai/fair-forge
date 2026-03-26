# Guardrail Models Evaluation Report

**Total Test Cases:** 185

## Overall Comparison

| Model                   | Accuracy   | Precision   | Recall   | F1 Score   |   Support | F1 (EN)   | F1 (ES)   |
|:------------------------|:-----------|:------------|:---------|:-----------|----------:|:----------|:----------|
| Qwen3Guard-0.6B         | 83.8%      | 100.0%      | 80.6%    | 89.3%      |       185 | 88.8%     | 90.3%     |
| Pipeline Multicapa      | 74.6%      | 95.0%       | 73.5%    | 82.9%      |       185 | 79.3%     | 89.1%     |
| GPT-OSS-SAFEGUARD-20B   | 88.1%      | 87.6%       | 100.0%   | 93.4%      |       185 | 94.1%     | 91.9%     |
| Llama-Guard-3-8B        | 69.7%      | 100.0%      | 63.9%    | 78.0%      |       185 | 77.6%     | 78.6%     |
| Granite-Guardian-3.1-2B | 78.9%      | 98.3%       | 76.1%    | 85.8%      |       185 | 88.3%     | 80.5%     |

## Recommendations

- **Best Overall:** GPT-OSS-SAFEGUARD-20B
- **Best for LATAM:** GPT-OSS-SAFEGUARD-20B
