import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from readme_generator.tools.github_pr_tool import GithubPRTool

LLM_BASE_URL = os.getenv("README_GENERATOR_LLM_BASE_URL", "http://10.54.34.78:30000/v1")
LLM_MODEL = os.getenv("README_GENERATOR_LLM_MODEL", "your-local-model")
LLM_API_KEY = os.getenv("README_GENERATOR_LLM_API_KEY", "empty")


@CrewBase
class GithubPRCrew:
    agents_config = "config/github_pr_agents.yaml"
    tasks_config = "config/github_pr_tasks.yaml"
    llm = LLM(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    def __init__(self, global_memory):
        self.global_memory = global_memory
        GithubPRTool.global_memory = global_memory

    @agent
    def github_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["github_agent"],
            tools=[
                GithubPRTool.get_publish_context,
                GithubPRTool.validate_publish_context,
                GithubPRTool.publish_family_artifacts,
                GithubPRTool.memory_store_pr_info,
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @task
    def github_pr(self) -> Task:
        return Task(config=self.tasks_config["github_pr_task"], agent=self.github_agent())

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True,
        )
