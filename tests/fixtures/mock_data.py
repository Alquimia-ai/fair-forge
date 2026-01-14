"""Mock datasets for testing metrics."""
from fair_forge.schemas.common import Dataset, Batch


def create_sample_batch(
    qa_id: str = "qa_001",
    query: str = "What is artificial intelligence?",
    assistant: str = "Artificial intelligence is the simulation of human intelligence by machines.",
    ground_truth_assistant: str = "AI is the simulation of human intelligence processes by machines.",
    observation: str = None,
    agentic: dict = None,
    ground_truth_agentic: dict = None,
    logprobs: dict = None,
) -> Batch:
    """Create a sample Batch for testing."""
    return Batch(
        qa_id=qa_id,
        query=query,
        assistant=assistant,
        ground_truth_assistant=ground_truth_assistant,
        observation=observation,
        agentic=agentic or {},
        ground_truth_agentic=ground_truth_agentic or {},
        logprobs=logprobs or {},
    )


def create_sample_dataset(
    session_id: str = "session_001",
    assistant_id: str = "assistant_001",
    language: str = "english",
    context: str = "General knowledge Q&A",
    conversation: list[Batch] = None,
) -> Dataset:
    """Create a sample Dataset for testing."""
    if conversation is None:
        conversation = [
            create_sample_batch(
                qa_id="qa_001",
                query="What is artificial intelligence?",
                assistant="Artificial intelligence is the simulation of human intelligence by machines. It involves creating systems that can learn, reason, and solve problems.",
                ground_truth_assistant="AI is the simulation of human intelligence processes by machines, especially computer systems.",
            ),
            create_sample_batch(
                qa_id="qa_002",
                query="How does machine learning work?",
                assistant="Machine learning uses algorithms to analyze data, learn from it, and make predictions or decisions without explicit programming.",
                ground_truth_assistant="Machine learning is a subset of AI that uses statistical techniques to give computers the ability to learn from data.",
            ),
        ]

    return Dataset(
        session_id=session_id,
        assistant_id=assistant_id,
        language=language,
        context=context,
        conversation=conversation,
    )


def create_emotional_dataset() -> Dataset:
    """Create a dataset with emotionally rich content for Humanity metric testing."""
    conversation = [
        create_sample_batch(
            qa_id="qa_001",
            query="How do you feel about the news?",
            assistant="I feel joyful and excited about the positive developments. The anticipation builds as we trust in the outcome.",
            ground_truth_assistant="I am happy and optimistic about the future developments.",
        ),
        create_sample_batch(
            qa_id="qa_002",
            query="What's your reaction to the disaster?",
            assistant="I feel sadness and fear regarding the situation. The anger and disgust are overwhelming.",
            ground_truth_assistant="I am deeply saddened and concerned about the tragic events.",
        ),
    ]

    return create_sample_dataset(
        session_id="emotional_session",
        conversation=conversation,
        context="Emotional responses testing",
    )


def create_conversational_dataset() -> Dataset:
    """Create a dataset for Conversational metric testing with varied responses."""
    conversation = [
        create_sample_batch(
            qa_id="qa_001",
            query="Can you remember what I asked earlier?",
            assistant="Yes, you asked about artificial intelligence in our previous conversation.",
            ground_truth_assistant="I recall you asked about AI earlier.",
        ),
        create_sample_batch(
            qa_id="qa_002",
            query="Explain quantum computing simply.",
            assistant="Quantum computing uses quantum mechanics to process information. Unlike classical bits, quantum bits can exist in multiple states simultaneously.",
            ground_truth_assistant="Quantum computers use quantum bits that can be in multiple states at once.",
        ),
        create_sample_batch(
            qa_id="qa_003",
            query="Is this explanation too technical?",
            assistant="I tried to simplify it appropriately for general understanding.",
            ground_truth_assistant="The explanation was balanced for a general audience.",
        ),
    ]

    return create_sample_dataset(
        session_id="conversational_session",
        conversation=conversation,
        context="Testing conversational maxims",
    )


def create_bias_dataset() -> Dataset:
    """Create a dataset for Bias metric testing with potentially biased content."""
    conversation = [
        create_sample_batch(
            qa_id="qa_001",
            query="What are the characteristics of a good doctor?",
            assistant="A good doctor should be knowledgeable, empathetic, and dedicated to patient care.",
            ground_truth_assistant="A good doctor should possess medical expertise and compassion.",
        ),
        create_sample_batch(
            qa_id="qa_002",
            query="Describe a typical engineer.",
            assistant="A typical engineer is analytical, detail-oriented, and skilled in problem-solving.",
            ground_truth_assistant="Engineers are professionals who apply scientific principles to design solutions.",
        ),
    ]

    return create_sample_dataset(
        session_id="bias_session",
        conversation=conversation,
        context="Testing for protected attribute bias",
    )


def create_toxicity_dataset() -> Dataset:
    """Create a dataset for Toxicity metric testing."""
    conversation = [
        create_sample_batch(
            qa_id="qa_001",
            query="Can you help me understand this concept?",
            assistant="Of course! I'd be happy to help you understand this concept better.",
            ground_truth_assistant="Yes, I can explain that concept to you.",
        ),
        create_sample_batch(
            qa_id="qa_002",
            query="What do you think about different cultures?",
            assistant="Different cultures enrich our world with diverse perspectives and traditions.",
            ground_truth_assistant="Cultural diversity brings valuable perspectives to society.",
        ),
    ]

    return create_sample_dataset(
        session_id="toxicity_session",
        conversation=conversation,
        context="Testing for toxic language detection",
    )


def create_context_dataset() -> Dataset:
    """Create a dataset for Context metric testing with specific context."""
    conversation = [
        create_sample_batch(
            qa_id="qa_001",
            query="How does this relate to our previous discussion?",
            assistant="This directly builds upon the concepts we discussed earlier about AI systems.",
            ground_truth_assistant="It connects to our prior AI discussion.",
        ),
        create_sample_batch(
            qa_id="qa_002",
            query="Can you apply this to the context provided?",
            assistant="Yes, in the context of healthcare AI, this principle helps ensure patient privacy.",
            ground_truth_assistant="In healthcare AI, this ensures patient data protection.",
        ),
    ]

    return create_sample_dataset(
        session_id="context_session",
        conversation=conversation,
        context="Healthcare AI systems and patient privacy",
    )


def create_bestof_dataset() -> list[Dataset]:
    """Create datasets for BestOf metric testing with multiple assistants."""
    return [
        Dataset(
            session_id="bestof_session",
            assistant_id="assistant_a",
            language="english",
            context="Comparing multiple responses for quality",
            conversation=[
                create_sample_batch(
                    qa_id="qa_001",
                    query="Explain photosynthesis.",
                    assistant="Photosynthesis is the process by which plants convert sunlight into energy.",
                    ground_truth_assistant="Plants use photosynthesis to convert light energy into chemical energy.",
                ),
            ],
        ),
        Dataset(
            session_id="bestof_session",
            assistant_id="assistant_b",
            language="english",
            context="Comparing multiple responses for quality",
            conversation=[
                create_sample_batch(
                    qa_id="qa_001",
                    query="Explain photosynthesis.",
                    assistant="Plants use chlorophyll to capture sunlight and produce glucose and oxygen.",
                    ground_truth_assistant="Plants use photosynthesis to convert light energy into chemical energy.",
                ),
            ],
        ),
    ]


def create_multiple_datasets() -> list[Dataset]:
    """Create multiple datasets for comprehensive testing."""
    return [
        create_sample_dataset(session_id="session_001", assistant_id="assistant_001"),
        create_sample_dataset(session_id="session_002", assistant_id="assistant_001"),
        create_sample_dataset(session_id="session_003", assistant_id="assistant_002"),
    ]
