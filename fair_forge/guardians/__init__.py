from fair_forge import Guardian
from fair_forge.schemas import ProtectedAttribute
from fair_forge.schemas import GuardianBias
from fair_forge.schemas import GuardianLLMConfig
from functools import partial

class IBMGranite(Guardian):
    """
    IBM Granite is a guardian that uses the IBM Granite model to detect bias in AI responses.
    
    This guardian analyzes conversations between users and AI assistants to identify potential
    biases related to protected attributes. It uses IBM's Granite model to make bias assessments
    and provides certainty scores for its determinations.

    Args:
        protected_attributes (list[ProtectedAttribute]): List of attributes to check for bias against
        config (GuardianLLMConfig): Configuration for the IBM Granite model including API settings
        **kwargs: Additional keyword arguments passed to the parent Guardian class

    Attributes:
        config (GuardianLLMConfig): The configuration used for the IBM Granite model
        provider: The configured model provider instance for making inference calls
    """
    def __init__(self,protected_attributes:list[ProtectedAttribute],config:GuardianLLMConfig,**kwargs):
        super().__init__(protected_attributes,**kwargs)
        self.config = config
        self.provider = config.provider(
            model=config.model,
            api_key=config.api_key,
            url=config.url,
            temperature=config.temperature,
            safe_token="Yes",
            unsafe_token="No"
        )

    def is_biased(self,question:str,answer:str,attribute:ProtectedAttribute,context:str) -> GuardianBias:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt = partial(
            self.tokenizer.apply_chat_template,
            conversation=messages,
            guardian_config={
                "risk_name": attribute.attribute.value,
                "risk_description": attribute.description,
            },
            tokenize=False,
            add_generation_prompt=True,
        )
        infer = self.provider.infer(prompt)
        return GuardianBias(
            is_biased=infer.is_bias,
            attribute=attribute.attribute.value,
            certainty=infer.probability
        )
        
class LLamaGuard(Guardian):
    """
    LLamaGuard is a guardian that uses the LLamaGuard model to detect bias in AI responses.
    
    This guardian implements bias detection using Meta's LLamaGuard model, which is specifically
    designed for content safety and bias detection. It analyzes conversations and provides
    assessments of potential biases related to protected attributes.

    Args:
        protected_attributes (list[ProtectedAttribute]): List of attributes to check for bias against
        config (GuardianLLMConfig): Configuration for the LLamaGuard model including API settings
        **kwargs: Additional keyword arguments passed to the parent Guardian class

    Attributes:
        config (GuardianLLMConfig): The configuration used for the LLamaGuard model
        provider: The configured model provider instance for making inference calls
    """
    def __init__(self,protected_attributes:list[ProtectedAttribute],config:GuardianLLMConfig,**kwargs):
        super().__init__(protected_attributes,**kwargs)
        self.config = config
        self.provider = config.provider(
            model=config.model,
            api_key=config.api_key,
            url=config.url,
            temperature=config.temperature,
            safe_token="safe",
            unsafe_token="unsafe"
        )

    def is_biased(self,question:str,answer:str,attribute:ProtectedAttribute,context:str) -> GuardianBias:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt = partial(
            self.tokenizer.apply_chat_template,
            conversation= messages,
            categories = {
                "S1": f"{attribute.attribute.value}.\n{attribute.description}"
            },
            tokenize= False,
            add_generation_prompt=True
        )
        infer = self.provider.infer(prompt)
        return GuardianBias(
            is_biased=infer.is_bias,
            attribute=attribute.attribute.value,
            certainty=infer.probability
        )