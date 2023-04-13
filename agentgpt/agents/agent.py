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
        self.tools=[]

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

        # Add the calculator tool to the agent
        self.add_tool(CalculatorTool())

    def add_tool(self, tool) -> None:
        """
        Add a tool to the agent.
        :param tool: The tool to add.
        :return: None
        """
        self.tools.append(tool)

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

    def ask_once(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a single response from the OpenAI ChatCompletion API
        given the system prompt and user prompt.

        :param system_prompt: The system prompt.
        :param user_prompt: The user prompt.
        :return: The response.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate a response using the OpenAI ChatCompletion API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            n=self.n,
            stop=self.stop,
            temperature=self.temperature,
        )

        # Extract the response text from the API response
        response_text = response.choices[0]["message"]["content"]

        return response_text


class AgentTool(object):
    """
    A tool that the agent can use to help it generate responses.
    """
    def __init__(self, command: str,
                 description: str,
                 usage: str):
        """
        Initialize the agent tool.
        :param command: The command to run the tool.
        :param description: A description of the tool.
        :param usage: A usage example of the tool.
        """
        self.command = command
        self.description = description
        self.usage = usage

    def __str__(self):
        """
        Return a string representation of the tool.
        :return: A string representation of the tool.
        """
        out = f"{self.command} - {self.description}\n\n"
        out += f"Usage Instructions:\n {self.usage}"

        return out

    def run(self, tool_input: str) -> str:
        """
        Run the tool.
        :param tool_input: The user input.
        :return: The tool's response.
        """
        raise NotImplementedError("The run method must be implemented by the tool.")


class CalculatorTool(AgentTool):
    """
    A tool that the agent can use to help evaluate arithmetic expressions.
    """
    def __init__(self):
        """
        Initialize the calculator tool.
        """
        self.command = "/calculate"
        self.description = "Evaluates the result of mathematical expressions."
        self.description += " If you want to add, subtract, multiply, or divide numbers, use this command."
        self.description += "instead of trying to do it yourself. Place this command at the last line of your output."
        self.description += "Do not attempt to calculate the expression yourself."
        self.usage = "Adding numbers: /calculate 2 + 2 + 3"
        self.usage += "Subtracting numbers: /calculate 2 - 2 - 3"
        self.usage += "Multiplying numbers: /calculate 2 * 2 * 3"
        self.usage += "Dividing numbers: /calculate 2 / 2 / 3"

    def run(self, tool_input: str) -> str:
        """
        Evaluate an arithmetic expression.
        :param tool_input: The input to evaluate.
        :return: The result of the evaluation.
        """
        # Evaluate the expression
        return str(eval(tool_input))


# class PythonEvaluatorTool()