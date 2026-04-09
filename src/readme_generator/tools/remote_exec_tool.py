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
        """Obtain a Python environment with SGLANG directly through the terminal by inputting the returned instruction string."""
        return "source /home/changrui/miniconda3/bin/activate && conda activate intel-extension"

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
    

