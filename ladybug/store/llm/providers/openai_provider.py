from typing import Dict, List, Optional, Union
from ..llm_interface import LLMInterface
from ..llm_enum import OpenAIEnum
from openai import OpenAI


class OpenAIProvider(LLMInterface):
    def __init__(self, 
        api_key: str, 
        api_url: str=None, 
        default_input_max_characters: int=1000, 
        default_generation_max_output_token: int=20,
        defult_generation_temperature: float=0,
    ):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_token = default_generation_max_output_token
        self.defult_generation_temperature = defult_generation_temperature

        self.generation_model_id = None
        self.enums = OpenAIEnum

        self.client = OpenAI(
            api_key=self.api_key, 
        )


    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
    

    def generate_text(
        self,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Union[str, int]]:
        
        messages = chat_history or []

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            temperature=self.defult_generation_temperature,
            max_tokens=self.default_generation_max_output_token
        )

        msg_content = response.choices[0].message.content
        usage = response.usage

        return {
            "text": msg_content,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }

    def construct_prompt(self, prompt: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": prompt[:self.default_input_max_characters].strip()}