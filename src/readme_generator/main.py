import sys
import time
from typing import List,Any,Optional,Dict

from crewai.flow.flow import Flow,listen,start
from pydantic import BaseModel

from crews.model_search_crew import ModelSearchCrew
from crews.readme_generate_crew import ReadmeGeneratorCrew
from crews.remote_execution_crew import RemoteExecutionCrew
from crews.readme_merger_crew import ReadmeMergerCrew
from crews.github_pr_crew import GithubPRCrew
from tools.memory_tool import GlobalMemory

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

class ModelWorkflowState(BaseModel):
    id: str = "model-workflow-default"
    model_list:List[str]=[]
    model_id_list:List[str]=[]
    model_url_list:List[str]=[]
    model_readme_list:List[str]=[]
    merged_readme:str=""
    reference_example:str=""
    merged_reference_example:str=""
    remote_folder:Optional[str]=None
    ssh_config:Dict[str,Any]={}
    github_config:Dict[str,Any]={}
    all_readmes_generated:bool=False
    all_test_completed:bool=False

class ModelWorkflowFlow(Flow[ModelWorkflowState]):
    initial_state=ModelWorkflowState

    def __init__(
        self,
        model_list:List[str],
        remote_folder:str,
        ssh_config:Optional[Dict[str,Any]],
        github_config:Optional[Dict[str,Any]],
        reference_example:str,
        merged_reference_example:str,
        **kwargs):
        super().__init__(**kwargs)
        self.state.model_list=model_list
        self.state.remote_folder=remote_folder
        self.state.ssh_config=ssh_config or {}
        self.state.github_config=github_config or {}
        self.state.reference_example=reference_example
        self.state.merged_reference_example=merged_reference_example
        self.global_memory=GlobalMemory()
        self.global_memory.memory_store("model_list",model_list)
        self.global_memory.memory_store("remote_folder",remote_folder)
        self.global_memory.memory_store("ssh_config",self.state.ssh_config)
        self.global_memory.memory_store("github_config",self.state.github_config)
        self.global_memory.memory_store("reference_example",self.state.reference_example)
        self.global_memory.memory_store("merged_reference_example",self.state.merged_reference_example)

    @start("wait_next_run")
    def run_model_search(self):
        print("\n启动模型搜索智能体")
        if not confirm_skip("是否跳过模型搜索？"):
            ModelSearchCrew().crew().kickoff()
        confirm_continue("是否进入 README 生成？")
        print("模型信息已存入全局内存")

    @listen(run_model_search)
    def run_readme_generation(self):
        print("\n启动README生成智能体")
        if not confirm_skip("是否跳过README生成？"):
            ReadmeGeneratorCrew().crew().kickoff()
        self.state.all_readmes_generated=True
        confirm_continue("是否执行远程测试？")

    @listen(run_readme_generation)
    def run_remote_execution(self):
        print("\n启动远程命令执行测试智能体")
        if not confirm_skip("是否跳过执行测试智能体？"):
            RemoteExecutionCrew().crew().kickoff()
        self.state.all_test_completed=True
        confirm_continue("是否合并 README？")

    @listen(run_remote_execution)
    def run_readme_merge(self):
        print("\n启动README合并智能体")
        if not confirm_skip("是否跳过README合并？"):
            ReadmeMergerCrew().crew().kickoff()
        confirm_continue("是否提交PR")

    @listen(run_readme_merge)
    def github_pr(self):
        print("\n提交github pr")
        if not confirm_skip("是否跳过提交github pr？"):
            GithubPRCrew().crew().kickoff()
        print("已提交PR")

    @listen(github_pr)
    def wait_next_run(self):
        print("\n 流程结束,等待60秒可重新运行...")
        time.sleep(60)

def kickoff():
    MODEL_LIST=["Llama-3.2-3B-Instruct","Llama-3.2-3B-quantized.w8a8"]
    REMOTE_FOLDER="/home/changrui"
    SSH_CONFIG={
        "hostname":"10.112.229.29",
        "port":22,
        "user_name":"changrui",
        "password":"Ycr2wy1314"
    }
    GITHUB_CONFIG={
        "token":"",
        "repo_owner":"",
        "repo_name":""
    }
    with open("reference_example.md","r",encoding="utf-8") as f:
        reference_example=f.read()
    with open("merged_reference_example.md","r",encoding="utf-8") as f:
        merged_reference_example=f.read()
    flow=ModelWorkflowFlow(
        model_list=MODEL_LIST,
        remote_folder=REMOTE_FOLDER,
        ssh_config=SSH_CONFIG,
        github_config=GITHUB_CONFIG,
        reference_example=reference_example,
        merged_reference_example=merged_reference_example
    )
    flow.kickoff()

if __name__=="__main__":
    kickoff()

        