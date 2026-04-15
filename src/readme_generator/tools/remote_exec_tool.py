import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
from crewai.llm import LLM
from .memory_tool import GlobalMemory
from .chatopenai import LLM_Callable
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
            stdin,stdout,stderr=self.ssh.exec_command(full_cmd,timeout=400)
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
    
class RemoteExecutionClient:
    def __init__(self):
        self.llm=LLM_Callable(
            base_url="http://10.54.34.78:30000/v1",
            api_key="empty",
            model_name="local-model"
        )
    
    def build_command(self,readme,option):
        """Extracts shell commands from README content based on the specified option.
        - If option is "build_sglang_env": Extract environment setup commands (clone, install, docker build).
        - If option is "test_model_building": Extract test commands (model download, server launch, benchmark).
        Input Parameters:
        - readme (str): Full README content text.
        - option (str): "build_sglang_env" or "test_model_building".
        
        Returns:
        - List[str]: Clean list of extracted shell commands, no thinking, no markdown, no explanations.
        """
        template="""
        You are an expert DevOps engineer who UNDERSTANDS technical documentation and GENERATES executable shell commands.

YOUR TASK:
1. READ and UNDERSTAND the full README content
2. EXTRACT all relevant context: repo URLs, branch names, Dockerfile names, model names, paths, etc.
3. GENERATE complete, executable shell commands based on the option
4. DO NOT just "extract" - INFER and COMPLETE commands if only references are present

STRICT RULES:
1. NO thinking, NO reasoning, NO explanations
2. NO markdown, NO code blocks, NO ```
3. ONE command PER LINE
4. ALL placeholders MUST be replaced with reasonable defaults or extracted values
5. If README mentions a repo but no clone command, GENERATE the git clone command
6. If README mentions a Dockerfile but no build command, GENERATE the docker build command
7. If README mentions a model but no download command, GENERATE the huggingface-cli download command
8. If NO commands can be generated, output EXACTLY: NO_COMMANDS_FOUND

README CONTENT:
{readme_content}

OPTION: {option}

WHAT TO GENERATE BASED ON OPTION:
- If option == "build_sglang_env":
  Generate commands for:
  - git clone (with correct URL and branch from README)
  - docker pull / docker build (with correct image name, Dockerfile from README)
  - pip install / conda install (inferred from README context)
  EXCLUDE: model download, server launch, benchmark

- If option == "test_model_building":
  Generate commands for:
  - huggingface-cli download (with model ID from README)
  - sglang serve / launch_server (with model path from README)
  - bench_serving / benchmark commands (inferred from README)
  EXCLUDE: initial environment setup

NOW GENERATE ONLY THE EXECUTABLE COMMANDS:"""
        prompt=template.format(readme_content=readme,option=option)
        command=self.llm.invoke(inputs=prompt).strip()
        import re
        command=re.sub(r"<think>.*?</think>","",command,flags=re.DOTALL)
        command = command.strip()

        if command == "NO_COMMANDS_FOUND":
            return []

        return [
            line.strip()
            for line in command.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    
    def detect_quantization_from_model_id(self,model_id):
        suffixes=["fp8","int8","awq","gptq","w8a8_int"]
        for suf in suffixes:
            if model_id.lower().encswith("w8a8"):
                return "w8a8_int"
            if model_id.lower().endswith(suf):
                return suf
        return None
    
    def fix_quantization_arg(self,commands,q_type):
        fixed=[]
        if type(commands)==str:
            commands=json.loads(commands)
        for cmd in commands:
            cmd=re.sub(r"--quantization\s+\S+","",cmd).strip()
            if q_type and q_type=="w8a8":
                cmd+="--quantization w8a8_int"
            if q_type:
                cmd+=f"--quantization {q_type}"
            fixed.append(cmd)
        return fixed 
    
    def fix_commands_by_llm(self,cmds:List[str],error_log:str)->List[str]:
        prompt = f"""
        You are a senior Linux & SGLANG deployment expert.
        Fix the following commands based on the error log.

        Original commands:
        {cmds}

        Error log:
        {error_log}

        Rules:
        1. Keep original command purpose and order.
        2. Fix only what is wrong.
        3. If missing SGLANG CPU env, add:
        cd /home/sdp/changrui && source .venv/bin/activate && export SGLANG_USE_CPU_ENGINE=1
        4. If permission denied: add sudo or chmod +x.
        5. If path not exist: add mkdir -p.
        6. If download failed: add --resume-download.
        7. Output ONLY the final command list, one per line.
        No extra words, no markdown, no explanation.
        """

        fixed_text=self.llm.invoke(prompt).strip()
        import re
        fixed_text=re.sub(r"<think>.*?</think>","",fixed_text,flags=re.DOTALL)
        fixed_text = fixed_text.strip()
        fixed_cmds=[
            line.strip()
            for line in fixed_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return fixed_cmds if fixed_cmds else cmds
    

class RemoteExecutionTool:
    client=RemoteExecutionClient()

    @tool("get_sglang_environment")
    def get_sglang_environment():
        """Get the official SGLANG CPU environment activation command.
        Returns: shell command string to activate SGLANG CPU environment.
        """
        return (
            "cd /home/sdp/changrui && source .venv/bin/activate && "
            "export SGLANG_USE_CPU_ENGINE=1 && "
            "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu && "
            "export LD_PRELOAD=/home/sdp/changrui/.venv/lib/libiomp5.so"
        )

    @tool("execute_on_remote_server")
    def execute_on_remote_server(commands:List[str],remote_folder:str)->str:
        """Execute shell commands on remote server via SSH.
        Automatically loads SSH_CONFIG from GLOBAL_MEMORY.
        Inputs: list of commands, remote working folder
        Returns: execution log string containing stdout, stderr, exit codes."""
        memory=GlobalMemory()
        ssh=memory.memory_retrieve("ssh_config")
        executor=RemoteGeneralExecutor(host=ssh["hostname"],user_name=ssh["user_name"],password=ssh["password"],
                                       port=ssh["port"])
        executor.connect()
        results=executor.execute_commands(
            command_list=commands,
            remote_folder=remote_folder
        )
        executor.disconnect()
        return results
    
    @tool("check_remote_model_exists")
    def check_remote_model_exists(model_id:str,remote_folder:str):
        """Check if the model exists in remote server's models folder.
        Inputs: model_id, remote_folder
        Returns: True if model exists, False otherwise."""
        from shlex import quote  # 安全转义路径，防止特殊字符/命令注入

        memory = GlobalMemory()
        ssh = memory.memory_retrieve("ssh_config")

        # 拼接完整模型路径（自动支持嵌套 model_id）
        model_full_path = f"{remote_folder}/models/{model_id}"
        # 安全转义，处理空格、/、.、-、: 等所有特殊字符
        safe_path = quote(model_full_path)

        # 安全判断远程目录是否存在（标准、兼容所有 Linux）
        cmd = f"test -d {safe_path} && echo EXISTS || echo NOT_EXISTS"

        executor = RemoteGeneralExecutor(
            host=ssh["hostname"],
            user_name=ssh["user_name"],
            password=ssh["password"],
            port=22
        )

        try:
            executor.connect()
            results = executor.execute_commands(
                command_list=[cmd],
                remote_folder=remote_folder
            )
            # 判断结果（兼容多行输出、空白字符）
            # 新代码（正确）
            output = results[0]["output"].strip()
            return "EXISTS" in output

        finally:
            # 确保无论是否异常，都断开连接
            executor.disconnect()
    
    @tool("download_model_from_huggingface")
    def download_model_from_huggingface(model_url:str,model_id:str,remote_folder:str):
        """ Download model from Hugging Face to remote server when model does not exist.
        Inputs: model_url (Hugging Face repo ID), remote_folder
        Returns: download execution log."""
        memory=GlobalMemory()
        ssh=memory.memory_retrieve("ssh_config")
        commands = [
            f"mkdir -p {remote_folder}/models",
            f"cd {remote_folder}/models",
            f"huggingface-cli download {model_url} --local-dir {model_id} --local-dir-use-symlinks False"
        ]
        executor=RemoteGeneralExecutor(host=ssh["hostname"],user_name=ssh["user_name"],password=ssh["password"],
                                       port=ssh["port"])
        executor.connect()
        res=executor.execute_commands(commands,remote_folder=remote_folder)
        executor.disconnect()
        return res

    @tool("build_command")
    def build_command(readme_str:str,option:str):
        """Extract shell commands from model_readme.
        Options: "build_sglang_env" (extract clone & docker build commands), "test_model_building" (extract test commands)
        Inputs: readme content, option
        Returns: list of extracted shell commands."""
        return RemoteExecutionTool.client.build_command(readme=readme_str,option=option)

    @tool("memory_retrieve_execution_context")
    def memory_retrieve_execution_context():
        """Retrieve all information needed for remote execution from GLOBAL_MEMORY.
        Returns: dictionary containing model_id_list, model_url_list, model_readme, github_url, ssh_config."""
        memory=GlobalMemory()
        return {
            "model_id_list":memory.memory_retrieve("model_id_list"),
            "model_url_list":memory.memory_retrieve("model_url_list"),
            "model_readme":memory.memory_retrieve("model_readme"),
            "github_url":memory.memory_retrieve("github_url"),
            "executed_command":memory.memory_retrieve("executed_command"),
            "execution_result":memory.memory_retrieve("execution_result"),
            "fail_reason_list":memory.memory_retrieve("fail_reason_list"),
            "ssh_config":memory.memory_retrieve("ssh_config"),
            "remote_folder":memory.memory_retrieve("remote_folder")
        }
    
    @tool("detect_quantization_from_model_id")
    def detect_quantization_from_model_id(model_id):
        """ Detect quantization type from model_id suffix.
        Input: model_id
        Returns: quantization type (e.g., "fp8", "int8", "int4") or None if no quantization."""
        return RemoteExecutionTool.client.detect_quantization_from_model_id(model_id=model_id)

    @tool("fix_command_quantization")
    def fix_command_quantization(commands,q_type):
        """Automatically fix --quantization parameter in commands.
        Inputs: list of commands, quantization type
        Returns: list of commands with corrected --quantization parameter."""
        return RemoteExecutionTool.client.fix_quantization_arg(commands=commands,q_type=q_type)

    @tool("memory_store_execution_result")
    def memory_store_execution_result(
        idx:int,
        command_str:str,
        result:str,
        fail_reason:Optional[str],
        updated_readme:Optional[str]
    ):
        """Store remote execution results into GLOBAL_MEMORY at specified index.
        Inputs: index, executed command, execution result, fail reason, updated readme
        Returns: success message."""
        def update_list(key,value):
            memory=GlobalMemory()
            lst=memory.memory_retrieve(key=key) or []
            while len(lst)<=idx:
                lst.append(None)
            lst[idx]=value
            memory.memory_store(key=key,value=lst)
        
        update_list("executed_command",command_str)
        update_list("execution_result",result)
        update_list("fail_reason_list",fail_reason)
        if updated_readme:
            update_list("model_readme",updated_readme)
        return True

    @tool("retry_allowed")
    def retry_allowed(current:int,max_retry=2):
        """Check if retry is allowed based on current retry count.
        Inputs: current retry count, max retry count (default 2)
        Returns: True if retry allowed, False otherwise."""
        return current<max_retry
    
    @tool("fix_command_errors")
    def fix_command_errors(cmds:List[str],err:str)->List[str]:
        """Use LLM to intelligently analyze remote execution error logs and automatically fix commands.
        Fixes: command not found, permission denied, path error, environment missing, library issues, etc.
        Inputs: list of original commands, error log
        Returns: list of fixed commands."""
        return RemoteExecutionTool.client.fix_commands_by_llm(cmds=cmds,error_log=err)
    
