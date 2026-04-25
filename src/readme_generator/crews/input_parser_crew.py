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
from readme_generator.tools.input_parser_tool import InputParseTool

LLM_BASE_URL = os.getenv("README_GENERATOR_LLM_BASE_URL", "http://10.54.34.78:30000/v1")
LLM_MODEL = os.getenv("README_GENERATOR_LLM_MODEL", "your-local-model")
LLM_API_KEY = os.getenv("README_GENERATOR_LLM_API_KEY", "empty")

@CrewBase
class InputParserCrew:
    agents_config="config/input_parser_agents.yaml"
    tasks_config="config/input_parser_tasks.yaml"

    llm=LLM(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

    def __init__(self, global_memory=None):
        self.global_memory = global_memory

    @agent
    def input_parser_agent(self)->Agent:
        memory_store_tool=InputParseTool.store_memory
        # memory_get_key_tool=InputParseTool.memory_get_keys
        # memory_get_value_type_tool=InputParseTool.memory_get_type
        ipnut_parse_text_tool=InputParseTool.parse_input_text
        get_input_text_tool=InputParseTool.get_input_text
        # set_github_config_tool=InputParseTool.set_github_config_to_memory
        # set_ssh_config_tool=InputParseTool.set_ssh_config_to_memory
        # set_remote_folder_tool=InputParseTool.set_remote_folder_to_memory
        # set_origin_reference_example_list_tool=InputParseTool.set_origin_reference_example_list_to_memory
        # set_merged_reference_example_tool=InputParseTool.set_merged_reference_example_to_memory

        return Agent(
            config=self.agents_config["input_parser_agent"],
            tools=[ipnut_parse_text_tool,memory_store_tool,get_input_text_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )
    
    @task
    def input_parse(self)->Task:
        return Task(config=self.tasks_config["input_parse_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True
        )
