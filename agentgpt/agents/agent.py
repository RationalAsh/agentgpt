"""
Base class for agents.
"""
import openai
import os
from enum import Enum, StrEnum

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about anything.
"""

DEFAULT_MODEL = "gpt-3.5-turbo"


# A StrEnum for all the possible roles in a conversation
class Role(StrEnum):
    # System
    SYSTEM = "system"
    # User
    USER = "user"
    # Assistant
    ASSISTANT = "assistant"


class Agent:
    """
    Base class for agents.
    """
    def __init__(self,
                 api_key: str = os.environ.get("OPENAI_API_KEY"),
                 organization_id: str = os.environ.get("OPENAI_ORGANIZATION"),
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = 4096,
                 temperature: float = 0.5,
                 top_p: float = 1.0,
                 n: int = 1,
                 stop: list = None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.api_key = api_key
        self.organization_id = organization_id
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.messages = []

        # Initialize OpenAI API
        openai.api_key = self.api_key
        openai.organization = self.organization_id

    def add_message(self, role: Role, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_chat_log(self) -> list:
        return self.messages

    def generate_response(self, user_input: str) -> str:
        self.add_message(Role.USER, user_input)

        # Generate a response using the OpenAI ChatCompletion API
        response = openai.ChatCompletion.create(
            model=self.model,
            prompt=[
                {"role": "system", "content": self.system_prompt},
                *self.messages,
            ],
            max_tokens=self.max_tokens,
            n=self.n,
            stop=self.stop,
            temperature=self.temperature,
        )

        # Extract the response text from the API response
        response_text = response.choices[0]["message"]["content"]

        # Add the generated response to the chat log
        self.add_message(Role.ASSISTANT, response_text)

        return response_text