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
from readme_generator.tools.remote_exec_tool import RemoteExecutionTool
from readme_generator.tools.get_step import create_step_callback

@CrewBase
class RemoteExecutionCrew:
    agents_config="config/remote_execute_agents.yaml"
    tasks_config="config/remote_execute_tasks.yaml"
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty" 
    )

    @agent 
    def remote_execution_agent(self)->Agent:
        remote_execution_tool=RemoteExecutionTool.execute_on_remote_server
        remote_sglang_tool=RemoteExecutionTool.get_sglang_environment
        remote_build_tool=RemoteExecutionTool.build_command
        remote_check_remote_model_exists_tool=RemoteExecutionTool.check_remote_model_exists
        remote_download_model_from_huggingface_tool=RemoteExecutionTool.download_model_from_huggingface
        remote_fix_command_quantization_tool=RemoteExecutionTool.fix_command_quantization
        remote_memory_store_execution_result_tool=RemoteExecutionTool.memory_retrieve_execution_context
        remote_retry_allowed_tool=RemoteExecutionTool.retry_allowed
        remote_fix_command_errors_tool=RemoteExecutionTool.fix_command_errors
        return Agent(
            config=self.agents_config["remote_execution_agent"],
            tools=[remote_build_tool,remote_sglang_tool,remote_execution_tool,remote_check_remote_model_exists_tool,remote_download_model_from_huggingface_tool,remote_fix_command_quantization_tool,remote_memory_store_execution_result_tool,remote_retry_allowed_tool,remote_fix_command_errors_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )
    
    @task
    def remote_execution(self)->Task:
        return Task(config=self.tasks_config["remote_execution_task"])

    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True
        )