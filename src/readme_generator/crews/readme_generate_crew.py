import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.llm import LLM
from readme_generator.tools.generate_readme_tool import GenerateReadmeTool

LLM_BASE_URL = os.getenv("README_GENERATOR_LLM_BASE_URL", "http://10.54.34.78:30000/v1")
LLM_MODEL = os.getenv("README_GENERATOR_LLM_MODEL", "your-local-model")
LLM_API_KEY = os.getenv("README_GENERATOR_LLM_API_KEY", "empty")


@CrewBase
class ReadmeGeneratorCrew:
    agents_config = "config/readme_generate_agents.yaml"
    tasks_config = "config/readme_generate_tasks.yaml"
    llm = LLM(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    def __init__(self, global_memory):
        self.global_memory = global_memory
        GenerateReadmeTool.global_memory = global_memory

    @agent
    def readme_generator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["readme_generator_agent"],
            tools=[
                GenerateReadmeTool.memory_retrieve_generation_context,
                GenerateReadmeTool.memory_store_family_artifacts,
                GenerateReadmeTool.memory_store_family_content,
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @task
    def adaptive_readme_generation_task(self) -> Task:
        return Task(
            config=self.tasks_config["adaptive_readme_generation_task"],
            agent=self.readme_generator_agent(),
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
