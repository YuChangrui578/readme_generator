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
        github_validate_pr_tool=GithubPRTool.validate_pr_exists_for_repo
        github_upload_pr_tool=GithubPRTool.upload_pr_for_repo
        github_config_tool=GithubPRTool.get_github_config
        github_merged_readme_tool=GithubPRTool.get_merged_readme
        github_info_store_tool=GithubPRTool.memory_store_pr_info
        return Agent(
            config=self.agents_config["github_agent"],
            tools=[github_validate_pr_tool,github_upload_pr_tool,github_config_tool,github_merged_readme_tool,github_info_store_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
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