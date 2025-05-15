class Batch(BaseModel):
    """
    A batch represents a single interaction between a user and an assistant in a conversation.
    Each batch contains the user's query, the assistant's response, and optional metadata.

    Attributes:
        qa_id (str): Unique identifier for the batch
        ground_truth_assistant (str): The expected or reference response from the assistant
        logprobs (Optional[dict]): Token-level log probabilities from the language model
        observation (Optional[str]): Any additional observations or notes about the interaction
        assistant (str): The actual response provided by the assistant
        query (str): The user's input or question
        agentic (Optional[dict]): Additional metadata or context about the agent's behavior
        ground_truth_agentic (Optional[dict]): Expected or reference metadata for the agent's behavior

    Example:
        ```python
        batch = Batch(
            qa_id="unique_id_123",
            query="What is the capital of France?",
            ground_truth_assistant="The capital of France is Paris.",
            assistant="Paris is the capital city of France.",
            logprobs={"tokens": ["Paris", "is", "the", "capital"], "probs": [0.9, 0.8, 0.7, 0.6]},
            observation="Assistant provided a clear and accurate response",
            agentic={"confidence": 0.95, "response_time": 1.2},
            ground_truth_agentic={"expected_confidence": 0.9, "expected_time": 1.0}
        )
        ```
    """
    ground_truth_assistant: str
    logprobs: Optional[dict] = None
    observation: Optional[str] = None
    assistant: str
    query: str
    agentic: Optional[dict] = None
    ground_truth_agentic: Optional[dict] = None
    qa_id: str

    class Config:
        """
        Pydantic model configuration.
        
        Attributes:
            arbitrary_types_allowed (bool): Allow arbitrary types in the model
            extra (str): How to handle extra fields ('forbid' means no extra fields allowed)
        """
        arbitrary_types_allowed = True
        extra = "forbid" 