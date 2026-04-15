import sys
import time
from typing import List,Any,Optional,Dict
import os

from crewai.flow.flow import Flow,listen,start
from pydantic import BaseModel

from crews.model_search_crew import ModelSearchCrew
from crews.readme_generate_crew import ReadmeGeneratorCrew
from crews.remote_execution_crew import RemoteExecutionCrew
from crews.readme_merger_crew import ReadmeMergerCrew
from crews.github_pr_crew import GithubPRCrew
from crews.input_parser_crew import InputParserCrew
from tools.memory_tool import GlobalMemory
from pathlib import Path
import json

def confirm_continue(message="continue?(y/n)"):
    choice=input(message).strip().lower()
    if choice!="y":
        print("stop!")
        sys.exit(0)

def confirm_skip(message="skip?(y/n)"):
    choice=input(message).strip().lower()
    if choice!="y":
        return False
    return True

def load_all_markdown_files(folder_path:str,recursive:bool=False)->List[str]:
    md_contents=[]
    folder=Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在:{folder_path}")
    
    pattern="**/*.md" if recursive else "*.md"
    for file_path in folder.glob(pattern):
        if file_path.suffix.lower() in [".md",".markdown"]:
            try:
                with open(file_path,"r",encoding="utf-8") as f:
                    content=f.read()
                    md_contents.append(content)
            except Exception as e:
                print(f"读取失败 {file_path}:{e}")
    return md_contents


class ModelWorkflowState(BaseModel):
    id: str = "model-workflow-default"
    model_list:List[str]=[]
    model_id_list:List[str]=[]
    model_url_list:List[str]=[]
    model_readme_list:List[str]=[]
    merged_readme:str=""
    reference_example_list:List[str]=[]
    origin_reference_example_list:List[str]=[]
    merged_reference_example:str=""
    remote_folder:Optional[str]=None
    ssh_config:Dict[str,Any]={}
    github_config:Dict[str,Any]={}
    all_readmes_generated:bool=False
    all_test_completed:bool=False
    github_url:List[str]=[]

class ModelWorkflowFlow(Flow[ModelWorkflowState]):
    initial_state=ModelWorkflowState

    def __init__(
        self,
        model_list:List[str],
        remote_folder:str,
        ssh_config:Optional[Dict[str,Any]],
        github_config:Optional[Dict[str,Any]],
        origin_reference_example_list:List[str],
        merged_reference_example:str,
        github_url:List[str],
        **kwargs):
        super().__init__(**kwargs)
        self.state.model_list=model_list
        self.state.remote_folder=remote_folder
        self.state.ssh_config=ssh_config or {}
        self.state.github_config=github_config or {}
        self.state.origin_reference_example_list=origin_reference_example_list
        self.state.merged_reference_example=merged_reference_example
        self.state.github_url=github_url
        self.global_memory=GlobalMemory()
        self.global_memory.memory_store("model_list",model_list)
        self.global_memory.memory_store("remote_folder",remote_folder)
        self.global_memory.memory_store("ssh_config",self.state.ssh_config)
        self.global_memory.memory_store("github_config",self.state.github_config)
        self.global_memory.memory_store("origin_reference_example_list",self.state.origin_reference_example_list)
        self.global_memory.memory_store("merged_reference_example",self.state.merged_reference_example)
        self.global_memory.memory_store("github_url",self.state.github_url)
        self.global_memory.save_to_file()

    @start("wait_next_run")
    def run_input_parser(self):
        print("\n启动输入解析智能体")
        if not confirm_skip("是否跳过输入解析？"):
            streaming_output=InputParserCrew().crew().kickoff()
            for chunk in streaming_output:
                if chunk.chunk_type=="text":
                    print(f"[{chunk.agent.role}]{chunk.content}",end="")
                elif chunk.chunk_type=="tool_use":
                    print(f"\n[工具调用]{chunk.tool_name}:{chunk.tool_input}")
        confirm_continue("是否进入模型搜索？")
    
    @listen(run_input_parser)
    def run_model_search(self):
        print("\n启动模型搜索智能体")
        if not confirm_skip("是否跳过模型搜索？"):
            streaming_output=ModelSearchCrew().crew().kickoff()
            for chunk in streaming_output:
                if chunk.chunk_type=="text":
                    print(f"[{chunk.agent.role}]{chunk.content}",end="")
                elif chunk.chunk_type=="tool_use":
                    print(f"\n[工具调用]{chunk.tool_name}:{chunk.tool_input}")
        confirm_continue("是否进入 README 生成？")
        print("模型信息已存入全局内存")

    @listen(run_model_search)
    def run_readme_generation(self):
        print("\n启动README生成智能体")
        if not confirm_skip("是否跳过README生成？"):
            streaming_output=ReadmeGeneratorCrew().crew().kickoff()
            for chunk in streaming_output:
                if chunk.chunk_type=="text":
                    print(f"[{chunk.agent.role}]{chunk.content}",end="")
                elif chunk.chunk_type=="tool_use":
                    print(f"\n[工具调用]{chunk.tool_name}:{chunk.tool_input}")
        self.state.all_readmes_generated=True
        confirm_continue("是否执行远程测试？")

    @listen(run_readme_generation)
    def run_remote_execution(self):
        print("\n启动远程命令执行测试智能体")
        if not confirm_skip("是否跳过执行测试智能体？"):
            streaming_output=RemoteExecutionCrew().crew().kickoff()
            for chunk in streaming_output:
                if chunk.chunk_type=="text":
                    print(f"[{chunk.agent.role}]{chunk.content}",end="")
                elif chunk.chunk_type=="tool_use":
                    print(f"\n[工具调用]{chunk.tool_name}:{chunk.tool_input}")
        self.state.all_test_completed=True
        confirm_continue("是否合并 README？")

    @listen(run_remote_execution)
    def run_readme_merge(self):
        print("\n启动README合并智能体")
        if not confirm_skip("是否跳过README合并？"):
            streaming_output=ReadmeMergerCrew().crew().kickoff()
            for chunk in streaming_output:
                if chunk.chunk_type=="text":
                    print(f"[{chunk.agent.role}]{chunk.content}",end="")
                elif chunk.chunk_type=="tool_use":
                    print(f"\n[工具调用]{chunk.tool_name}:{chunk.tool_input}")
        confirm_continue("是否提交PR")

    @listen(run_readme_merge)
    def github_pr(self):
        print("\n提交github pr")
        if not confirm_skip("是否跳过提交github pr？"):
            streaming_output=GithubPRCrew().crew().kickoff()
            for chunk in streaming_output:
                if chunk.chunk_type=="text":
                    print(f"[{chunk.agent.role}]{chunk.content}",end="")
                elif chunk.chunk_type=="tool_use":
                    print(f"\n[工具调用]{chunk.tool_name}:{chunk.tool_input}")
        print("已提交PR")

    @listen(github_pr)
    def wait_next_run(self):
        print("\n 流程结束,等待60秒可重新运行...")
        time.sleep(60)

def kickoff():
    # MODEL_LIST=["Llama-3.2-3B-Instruct","Llama-3.2-3B-quantized.w8a8","Llama-3.2-3B-Instruct-AWQ"]
    MODEL_LIST=["Llama-3.2-3B-quantized.w8a8","Llama-3.2-3B-Instruct-FP8","Llama-3.2-3B-Instruct-AWQ"]
    REMOTE_FOLDER="/home/sdp/changrui"
    SSH_CONFIG={
        "hostname":"10.165.58.104",
        "port":22,
        "user_name":"sdp",
        "password":"$harktank2Go"
    }
    GITHUB_CONFIG={
        "github_token":"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",
        "repo_owner":"YuChangrui578",
        "repo_name":"readme_example",
        "base_branch":"main",
        "head_branch":"dev",
        "pr_title":"test",
        "pr_description":"test_github_pr",
        "commit_message":"test",
        "path":"Xeon/Llama/Llama-3.2-3B-Instruct.md"
    }
    folder_path="/home/changrui/readme_generator/src/readme_generator/reference_example"
    origin_reference_example_list=load_all_markdown_files(folder_path=folder_path)
    github_url=["","","https://github.com/jianan-gu/sglang/tree/cpu_optimized"]
    # github_url=[""]
    with open("/home/changrui/readme_generator/src/readme_generator/merged_reference_example.md","r",encoding="utf-8") as f:
        merged_reference_example=f.read()
    flow=ModelWorkflowFlow(
        model_list=MODEL_LIST,
        remote_folder=REMOTE_FOLDER,
        ssh_config=SSH_CONFIG,
        github_config=GITHUB_CONFIG,
        origin_reference_example_list=origin_reference_example_list,
        merged_reference_example=merged_reference_example,
        github_url=github_url
    )
    flow.kickoff()

if __name__=="__main__":
    kickoff()

        