import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# 将根路径添加到 sys.path，让 Python 能识别 readme_generator 模块
if root_path not in sys.path:
    sys.path.append(root_path)
from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from readme_generator.tools.github_pr_tool import GithubPRTool
from readme_generator.tools.memory_tool import MemoryTool
from readme_generator.tools.get_step import create_step_callback


@CrewBase
class GithubPRCrew:
    agents_config="config/github_pr_agents.yaml"
    tasks_config="config/github_pr_tasks.yaml"
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )

    @agent 
    def github_agent(self)->Agent:
        github_create_pr_tool=GithubPRTool.create_new_pr_for_repo
        github_validate_pr_tool=GithubPRTool.validate_pr_exists_for_repo
        github_upload_pr_tool=GithubPRTool.upload_pr_for_repo
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        memory_get_key_tool=MemoryTool.get_memory_key
        return Agent(
            config=self.agents_config["github_agent"],
            tools=[github_create_pr_tool,github_validate_pr_tool,github_upload_pr_tool,memory_store_tool,memory_retrieve_tool,memory_get_key_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            #step_callback=create_step_callback(agent_name="github_agent")
        )
    
    @task 
    def github_pr(self)->Task:
        return Task(config=self.tasks_config["github_pr_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True
        )