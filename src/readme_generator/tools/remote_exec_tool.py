import os
import time
from typing import List,Dict,Optional,Any
from pydantic import BaseModel,Field
from crewai.tools import tool
import re
import paramiko
import tempfile
import sys
import json
import select
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class RemoteGeneralExecutor:
    def __init__(
        self,
        host:str,
        user_name:str,
        password:Optional[str]=None,
        key_filename:Optional[str]=None,
        port:int=22,
    ):
        self.host=host
        self.user_name=user_name
        self.password=password
        self.key_filename=key_filename
        self.port=port
        self.ssh=None
    
    def connect(self):
        self.ssh=paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_args={
            "hostname":self.host,
            "username":self.user_name,
            "port":self.port,
            "timeout":15
        }
        if self.key_filename:
            connect_args["key_filename"]=os.path.expanduser(self.key_filename)
        else:
            connect_args["password"]=self.password
        self.ssh.connect(**connect_args)
        print("✅ SSH 连接成功")

    def disconnect(self):
        if self.ssh:
            try:
                self.ssh.close()
            finally:
                self.ssh=None   
    
    def execute_commands(self,command_list:List[str],remote_folder:str)->List[Dict[str,Any]]:
        if not self.ssh:
            raise Exception("请先调用connect()建立ssh连接")
        results=[]
        remote_folder=remote_folder.strip()
        for cmd in command_list:
            full_cmd=f"cd {remote_folder} && {cmd}"
            stdin,stdout,stderr=self.ssh.exec_command(full_cmd,timeout=3600)
            output=stdout.read().decode("utf-8",errors="ignore")
            error=stderr.read().decode("utf-8",errors="ignore")
            exit_code=stdout.channel.recv_exit_status()
            results.append({
                "command":cmd,
                "full_command":full_cmd,
                "remote_folder":remote_folder,
                "output":output,
                "error":error,
                "exit_code":exit_code
            })
        return results
    
    

class RemoteExecutionTool:

    @tool("Execute SSH")
    def execute_on_remote_server(commands:List[str],host:str,user_name:str,password:str,remote_folder:str,key_filename:Optional[str]=None)->str:
        """Executes a list of shell commands on a configured remote GPU server via SSH.\
        It automatically handles connection, navigation to the working directory, execution, and output capture.\
        YOU (LLM) must provide the commands clearly. \
        Returns a detailed log of stdout, stderr, and exit codes."""
        import pdb;pdb.set_trace()
        executor=RemoteGeneralExecutor(host=host,user_name=user_name,password=password,
                                       key_filename=key_filename)
        executor.connect()
        results=executor.execute_commands(
            command_list=commands,
            remote_folder=remote_folder
        )
        executor.disconnect()
        return results
    
    @tool("Get SGLANG Environment")
    def get_sglang_environment():
        """Obtain a Python environment with SGLANG directly through the terminal by inputting the returned instruction string if not github_url"""
        return """cd /home/sdp/changrui && source .venv/bin/activate && export SGLANG_USE_CPU_ENGINE=1 && export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu && export ${LD_PRELOAD}:/home/sdp/changrui/.venv/lib/libiomp5.so:${LD_LIBRARY_PATH}/libtcmalloc.so.4:${LD_LIBRARY_PATH}/libtbbmalloc.so.2"""

    @tool("Build Commands")
    def build_command(readme_str:str,option:str):
        """Extracts commands related to building or testing from the provided README content, adapting them based on the specified operation mode (option).

  Input Parameters:
  - readme_content (str): The full text content of the README file.
  - option (str): Specifies the context for command extraction. Valid values are "build_sglang_env" or "test_model_building".
    - If "build_sglang_env": Extracts commands for cloning repositories, installing dependencies, building Docker images, or configuring the environment.
    - If "test_model_building": Extracts commands for downloading models, launching services (sglang server), and running benchmarks.

  Functional Logic:
  1. Parses code blocks and instruction paragraphs in the README.
  2. Filters irrelevant commands based on the option (e.g., ignores pure environment installation steps in test mode unless they are strict prerequisites).
  3. Preliminarily standardizes command formats (e.g., handling newlines, variable placeholders).

  Output:
  - A list or string of extracted and preliminarily processed commands. 
  - Note: This tool is responsible only for "extraction." It does not execute commands or perform final business logic corrections (such as adding quantization parameters or checking paths); these corrections are handled by the Agent in the main workflow."""
        template="""
        You are an expert DevOps engineer specializing in extracting executable shell commands from technical documentation.

**Task:**
Analyze the provided README content and extract specific shell commands based on the specified `option`.

**Input Parameters:**
- **README Content**: {readme_content}
- **Option**: {option} (Value is either "build_sglang_env" or "test_model_building")

**Extraction Logic based on Option:**

1. **If option is "build_sglang_env"**:
   - Extract commands related to:
     - Cloning repositories (git clone).
     - Installing system dependencies (apt-get, yum, etc.).
     - Installing Python dependencies (pip install, conda install).
     - Building Docker images (docker build) or setting up Conda environments.
   - **Exclude**: Model downloading, model serving, or benchmarking commands.

2. **If option is "test_model_building"**:
   - Extract commands related to:
     - Downloading model weights (huggingface-cli, wget, etc.).
     - Launching the model server (e.g., python -m sglang.launch_server, torchrun, etc.).
     - Running benchmarks or inference tests (e.g., bench_serving, curl requests for testing).
   - **Exclude**: General environment setup steps (like initial pip install or docker build) unless they are strictly required immediately before the test in a single script context.

**General Constraints:**
- **Preserve Placeholders**: Keep variables like `<model_path>`, `<port>`, or `$VAR` as they appear.
- **Command Chaining**: If multiple sequential steps are required for the specific option, combine them using `&&` or output them as separate lines if they represent distinct phases.
- **Clean Output**: Remove markdown formatting, comments, or explanatory text from the commands themselves.

**Output Requirement:**
- Output **ONLY** the raw command string(s).
- Do **NOT** include markdown code blocks (```), explanations, headers, or any additional text.
- If no relevant commands are found for the given option, output "NO_COMMANDS_FOUND".
"""
        prompt_template=PromptTemplate(template=template)
        retrieval_chains={}
        parallel_tasks=retrieval_chains.copy()
        parallel_tasks["input"]=RunnablePassthrough()
        qa_chain=(
            RunnableParallel(parallel_tasks)
            |RunnableLambda(lambda x:{
                "readme_content":x["input"],
                "option":option
            })
            |prompt_template
            |StrOutputParser()
        )
        command=qa_chain.invoke(input=readme_str)
        import pdb;pdb.set_trace(command)
        return command
    # @tool("Extract SSH")
    # def extract_shell_commands(markdown_content:str)->List[Dict[str,str]]:
    #     """Extract the relevant terminal commands from the README document."""
    #     pattern = r"```(\w+)?\n(.*?)```"
    #     matches=re.findall(pattern,markdown_content,re.DOTALL)
    #     commands=[]
    #     shell_languages=["bash","sh","shell","cmd","console","zsh"]
    #     for lang,code in matches:
    #         lang_lower=(lang or "").lower()
    #         if lang_lower in shell_languages:
    #             clean_code = re.sub(r'^[\$\>]\s*', '', code, flags=re.MULTILINE)
    #             if clean_code.strip() and not clean_code.strip().startwith("#"):
    #                 commands.append({
    #                     "language":lang_lower,
    #                     "code":clean_code.strip()
    #                 })
    #     return commands
    
    # @tool("Build commands")
    # def build_final_commands(
    #     command_templates:List[str],
    #     model_id:str,
    #     remote_folder:str
    # )->List[str]:
    #     """
    #     Input the original command list and automatically replace all variables:
    #     <MODEL_ID>
    #     <MODEL_ID_OR_PATH>
    #     <path/to/local/dir>
    #     """
    #     final_commands = []
    #     for cmd in command_templates:
    #         # 替换所有变量
    #         cmd = cmd.replace("<MODEL_ID>", model_id)
    #         cmd = cmd.replace("<MODEL_ID_OR_PATH>", model_id)
    #         cmd = cmd.replace("<path/to/local/dir>", remote_folder)
    #         # 清理多余空格、换行（保证命令可执行）
    #         cmd = " ".join(cmd.strip().split())
    #         final_commands.append(cmd)
    #     return final_commands
    

