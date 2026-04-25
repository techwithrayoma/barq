from ladybug.core.config import Settings
from .llm_enum import LLMEnum
from .providers.openai_provider import OpenAIProvider

class LLMProviderFactory:
    def __init__(self, config: Settings):
        self.config = config
    
    def create(self, provider: str):

        if provider == LLMEnum.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_BASE_URL,
                default_input_max_characters=self.config.OPENAI_MAX_INPUT_CHARS,
                default_generation_max_output_token=self.config.OPENAI_MAX_OUTPUT_TOKENS,
                defult_generation_temperature=self.config.OPENAI_TEMPERATURE
            )
        
        return None 