"""
Clone of babyagi by yoheinakajima
"""

from agentgpt.agents.agent import *
from typing import List
from enum import Enum

TASK_CREATOR_SYSTEM_PROMPT = """
You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: 
{objective}, 

The last completed task has the result: {result}.

This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
Return the tasks as an array.
"""

INITIAL_TASK_LIST = [
    "Create a list of tasks to achieve the objective."
]

class TaskStatus(str, Enum):
    """
    Task status.
    """
    TODO = "TODO"
    DONE = "DONE"
    FAIL = "FAIL"

class Task(object):
    """
    Task Object.
    """
    def __init__(self, description: str,
                 status: TaskStatus=TaskStatus.TODO):
        self.description = description
        self.status = TaskStatus.TODO

    def __str__(self):
        return f"[{self.status}] {self.description}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.description == other.description and self.status == other.status

    def set_done(self):
        self.status = TaskStatus.DONE

    def set_fail(self):
        self.status = TaskStatus.FAIL

    def is_done(self):
        return self.status == TaskStatus.DONE

class BabyAGI(Agent):
    """
    BabyAGI agent.
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
        self.tools = [PythonEvalTool]

    def execute_task(self, task: str) -> str:
        """
        Execute the tasks.
        :return:
        """
        tool_descriptions = '\n'.join([f"{t}" for t in self.tools])
        system_prompt = f"""
        You are task execution agent that systematically carries out the current task
        to the best of your ability so as to achieve a stated objective.
        
        If a task requires precise outputs or inputs, you can make use of commands
        to help you execute the task. Output the command and arguments as specified
        by the tool descriptions to use the tool.
        """
        user_prompt = f"""
        Objective: {self.objective}
        Current Task: {task}
        
        Tools Available:
        {tool_descriptions}
        
        Think through this step by step. Show your thinking process first
        under a heading called "Thinking Process". If you need the output from a tool,
        use the appropriate tool command and place it under the heading "Action:" and wait 
        for a reply. Do not output anything else until you get a reply.
        """

        self.logger.info("Executing task: %s", task)

        response = self.ask_once(system_prompt=system_prompt,
                                 user_prompt=user_prompt)

        return response


if __name__ == '__main__':
    babyagi = BabyAGI(objective="Add the numbers in this list: 3.141, 11.5, 11.5, 11.5, 11.5.",
                      task_description="Create a list of tasks to achieve the objective.",
                      task_list=INITIAL_TASK_LIST)

    resp = babyagi.execute_task("Add the numbers.")
    i = resp.find('/calculate')
    if i != -1:
        j = resp.find('\n', i)

    print(resp)