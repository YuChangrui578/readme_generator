import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# 将根路径添加到 sys.path，让 Python 能识别 readme_generator 模块
if root_path not in sys.path:
    sys.path.append(root_path)
    
from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from readme_generator.tools.memory_tool import MemoryTool
from readme_generator.tools.get_step import create_step_callback
from readme_generator.tools.merge_readme_tool import MergeReadmeTool

@CrewBase
class ReadmeMergerCrew:
    agents_config="config/readme_merge_agents.yaml"
    tasks_config="config/readme_merge_tasks.yaml"
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )
    @agent
    def merge_readme_agent(self)->Agent:
        memory_store_tool=MergeReadmeTool.memory_store_merged_readme
        memory_retrieve_tool=MergeReadmeTool.memory_retrieve_merge_context
        memory_get_key_tool=MemoryTool.get_memory_key
        merge_model_series_readme_tool=MergeReadmeTool.merge_model_series_readme
        return Agent(
            config=self.agents_config["merge_readme_agent"],
            tools=[merge_model_series_readme_tool,memory_store_tool,memory_get_key_tool,memory_retrieve_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True, 
        )
    
    @task
    def readme_merge(self)->Task:
        return Task(config=self.tasks_config["readme_merge_task"])

    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True
        )