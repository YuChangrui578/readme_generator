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
from readme_generator.tools.generate_readme_tool import GenerateReadmeTool
from readme_generator.tools.get_step import create_step_callback

@CrewBase
class ReadmeGeneratorCrew:
    agents_config="config/readme_generate_agents.yaml"
    tasks_config="config/readme_generate_tasks.yaml"
    # llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )

    @agent
    def readme_generator_agent(self)->Agent:
        memory_store_tool=GenerateReadmeTool.memory_store_model_readme
        memory_retrieve_tool=GenerateReadmeTool.memory_retrieve_model_all_info
        batch_generate_model_readme_tool=GenerateReadmeTool.batch_generate_model_readme
        get_reference_example=GenerateReadmeTool.get_reference_example_list
        return Agent(
            config=self.agents_config["readme_generator_agent"],
            llm=self.llm,
            tools=[memory_store_tool,memory_retrieve_tool,batch_generate_model_readme_tool,get_reference_example],
            verbose=True,
            allow_delegation=True,
        )
    
    @task
    def readme_generate(self)->Task:
        return Task(config=self.tasks_config["readme_generate_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True
        )