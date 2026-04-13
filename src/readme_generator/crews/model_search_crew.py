import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# 将根路径添加到 sys.path，让 Python 能识别 readme_generator 模块
if root_path not in sys.path:
    sys.path.append(root_path)

from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from readme_generator.tools.model_search_tool import ModelSearchTool
from readme_generator.tools.memory_tool import MemoryTool
from readme_generator.tools.get_step import create_step_callback
from langchain_openai import ChatOpenAI

@CrewBase
class ModelSearchCrew:
    agents_config="config/model_search_agents.yaml"
    tasks_config="config/model_search_tasks.yaml"
    # llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )
    # llm = LLM(
    #     model="your-local-model",
    #     base_url="http://10.112.229.29:30000/v1",
    #     api_key="empty"
    # )

    @agent
    def model_search_agent(self)->Agent:
        model_search_tool=ModelSearchTool.huggingface_model_search_url
        #model_search_mirror_tool=ModelSearchTool.huggingface_mirror_model_search_url
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        memory_get_key_tool=MemoryTool.get_memory_key
        return Agent(
            config=self.agents_config["model_search_agent"],
            tools=[model_search_tool,memory_store_tool,memory_retrieve_tool,memory_get_key_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
            step_callback=create_step_callback(agent_name="model_search_agent")
        )
    
    @task
    def model_search(self)->Task:
        return Task(config=self.tasks_config["model_search_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True
        )