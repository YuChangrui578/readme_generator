import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from readme_generator.tools.remote_exec_tool import RemoteExecutionTool

LLM_BASE_URL = os.getenv("README_GENERATOR_LLM_BASE_URL", "http://10.54.34.78:30000/v1")
LLM_MODEL = os.getenv("README_GENERATOR_LLM_MODEL", "your-local-model")
LLM_API_KEY = os.getenv("README_GENERATOR_LLM_API_KEY", "empty")


@CrewBase
class RemoteExecutionCrew:
    agents_config = "config/remote_execute_agents.yaml"
    tasks_config = "config/remote_execute_tasks.yaml"
    llm = LLM(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    def __init__(self, global_memory):
        self.global_memory = global_memory
        RemoteExecutionTool.global_memory = global_memory

    @agent
    def remote_execution_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["remote_execution_agent"],
            tools=[
                RemoteExecutionTool.memory_retrieve_execution_context,
                RemoteExecutionTool.memory_preview_remote_content,
                RemoteExecutionTool.execute_remote_readme_validation,
                RemoteExecutionTool.memory_store_execution_result,
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @task
    def adaptive_remote_validation_task(self) -> Task:
        return Task(
            config=self.tasks_config["adaptive_remote_validation_task"],
            agent=self.remote_execution_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True,
        )
