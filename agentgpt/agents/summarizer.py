"""
A summarizer agent.
"""

from typing import List
from agentgpt.agents.agent import *
import argparse
from pdfminer.high_level import extract_text

SUMMARIZER_SYSTEM_PROMPT = """
You are a summarizer that can summarize any text. Your goal is to keep 
the most important information in the text while removing the rest and 
keeping the text as short as possible. The summary does not have to be
grammatically correct as long as it is generally understandable.
"""

class Summarizer(Agent):
    """
    Summarizer agent.
    """
    def __init__(self,
                 api_key: str = os.environ.get("OPENAI_API_KEY"),
                 organization_id: str = os.environ.get("OPENAI_ORGANIZATION"),
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = 2048,
                 temperature: float = 0.5,
                 top_p: float = 1.0,
                 n: int = 1,
                 stop: List[str] = None,
                 system_prompt: str = SUMMARIZER_SYSTEM_PROMPT):
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
        super().__init__(api_key=api_key,
                         organization_id=organization_id,
                         model=model,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         top_p=top_p,
                         n=n,
                         stop=stop,
                         system_prompt=system_prompt)

    def summarize(self, text: str) -> str:
        """
        Summarize text.
        :param text: The text to summarize.
        :return: The summary.
        """
        # User prompt
        user_prompt = f"""Please summarize the following text for me: {text}\n\n"""
        user_prompt += f"Be concise and respond with only the summary and nothing else."

        return self.generate_one_shot(text)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Specify that two arguments are mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text_file", type=str, help="Text file to summarize.")
    group.add_argument("--pdf_file", type=str, help="PDF file to summarize.")

    # Parse arguments
    args = parser.parse_args()

    # Initialize summarizer
    summarizer = Summarizer()

    # Summarize text
    if args.text_file:
        with open(args.text_file, "r") as f:
            text = f.read()
    else:
        text = extract_text(args.pdf_file)

    # print(text)
    summary = summarizer.summarize(text)
    print(summary)
