"""
Task Creation Agent for BabyAGI by yoheinakajima
"""

from agentgpt.agents.agent import *
from typing import List

TASK_CREATOR_SYSTEM_PROMPT = """
You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: 
{objective}, 

The last completed task has the result: {result}.

This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
Return the tasks as an array.
"""

class TaskCreator(Agent):
    """
    Task creator agent.
    """
    def __init__(self,
                 objective: str,
                 task_description: str,
                 task_list: List[str],
                 api_key: str = os.environ.get("OPENAI_API_KEY"),
                 organization_id: str = os.environ.get("OPENAI_ORGANIZATION"),
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = 2048,
                 temperature: float = 0.5,
                 top_p: float = 1.0,
                 n: int = 1,
                 stop: List[str] = None,
                 system_prompt: str = TASK_CREATOR_SYSTEM_PROMPT):
        """
        Initialize the agent.
        :param objective: The objective of the task.
        :param task_description: The task description.
        :param task_list: Initial list of tasks.
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
        self.objective = objective
        self.task_description = task_description
        self.task_list = task_list

    def create_task(self, text: str) -> str:
        """
        Create a task.
        :param text: The text to summarize.
        :return: The summary.
        """
        # User prompt
        user_prompt = f"""Please create a task for me based on the following text: {text}\n\n"""
        # Generate
        return self.generate_response(user_input=user_prompt)