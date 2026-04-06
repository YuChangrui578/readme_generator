import os
import time
from typing import List,Dict
from crewai_tools import BaseTool
from pydantic import BaseModel,Field
from langchain_tools import tool

class RemoteExecInput(BaseModel):
    commands:List[str]=Field(...,description="List of shell commands to execute on the remote server.")
    model_name:str=Field(..., description="Model name for logging context.")
    setup_steps:List[str]=Field(default=[],description="Optional setup commands (e.g., cd directory, activate venv) to run before main commands.")

class RemoteExecutionTool():

    @tool("Executes a list of shell commands on a configured remote GPU server via SSH.\
        It automatically handles connection, navigation to the working directory, execution, and output capture.\
        YOU (LLM) must provide the commands clearly. \
        Returns a detailed log of stdout, stderr, and exit codes.")
    def execute_on_remote_server(commands:List[str],model_name:str,setup_steps:List[str]=None)->str:
        pass
