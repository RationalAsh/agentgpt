"""
Base class for agents.
"""
import openai
import os
from enum import Enum
import logging

DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about anything.
"""

DEFAULT_MODEL = "gpt-3.5-turbo"


# A StrEnum for all the possible roles in a conversation
class Role(str, Enum):
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
                 max_tokens: int = 2048,
                 temperature: float = 0.5,
                 top_p: float = 1.0,
                 n: int = 1,
                 stop: list = None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        """
        Initialize the agent.
        :param api_key: OpenAI API key
        :param organization_id: OpenAI organization ID
        :param model: OpenAI model
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: Model temperature.
        :param top_p: Model top_p parameter.
        :param n: Number of responses to generate.
        :param stop: List of tokens to stop generating on.
        :param system_prompt: The system prompt.
        """
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

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Add a handler to log to a file
        handler = logging.StreamHandler()
        # Add a formatter to the handler
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger.addHandler(handler)
        # Set a prefix for all log messages

        # Log the agent configuration
        self.logger.info("Initializing agent...")
        self.logger.info(f"Agent configuration: {self.__dict__}")

    def add_message(self, role: Role, content: str) -> None:
        """
        Add a message to the chat log.
        :param role: The role of the message.
        :param content: The content of the message.
        :return: None
        """
        self.messages.append({"role": role, "content": content})

    def get_chat_log(self) -> list:
        """
        Get the chat log.
        :return: The chat log.
        """
        return self.messages

    def generate_one_shot(self, user_input: str) -> str:
        """
        Generate a one-shot response to the user input without
        adding the user input to the chat log.
        :param user_input: The user input
        :return: The generated response
        """

        # Generate a response using the OpenAI OneShotCompletion API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
        )

        # Extract the response text from the API response
        response_text = response.choices[0]["message"]["content"]

        return response_text

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to the user input and add the user input
        :param user_input: The user input
        :return: The generated response
        """
        self.add_message(Role.USER, user_input)

        # Generate a response using the OpenAI ChatCompletion API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
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