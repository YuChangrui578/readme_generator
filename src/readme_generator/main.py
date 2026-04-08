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

class ModelWorkflowState(BaseModel):
    id: str = "model-workflow-default"
    model_list:List[str]=[]
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
        **kwargs):
        super().__init__(**kwargs)
        self.state.model_list=model_list
        self.state.remote_folder=remote_folder
        self.state.ssh_config=ssh_config or {}
        self.state.github_config=github_config or {}
        self.global_memory=GlobalMemory()
        self.global_memory.memory_store("model_list",model_list)
        self.global_memory.memory_store("remote_folder",remote_folder)
        self.global_memory.memory_store("ssh_config",self.state.ssh_config)
        self.global_memory.memory_store("github_config",self.state.github_config)

    @start("wait_next_run")
    def run_model_search(self):
        print("\n启动模型搜索智能体")
        ModelSearchCrew().crew().kickoff()
        import pdb;pdb.set_trace()
        print("模型信息已存入全局内存")

    @listen(run_model_search)
    def run_readme_generation(self):
        print("\n启动README生成智能体")
        import pdb;pdb.set_trace()
        ReadmeGeneratorCrew().crew().kickoff()
        import pdb;pdb.set_trace()
        self.state.all_readmes_generated=True

    @listen(run_readme_generation)
    def run_remote_execution(self):
        print("\n启动远程命令执行测试智能体")
        import pdb;pdb.set_trace()
        RemoteExecutionCrew().crew().kickoff()
        import pdb;pdb.set_trace()
        self.state.all_test_completed=True

    @listen(run_remote_execution)
    def run_readme_merge(self):
        print("\n启动README合并智能体")
        import pdb;pdb.set_trace()
        ReadmeMergerCrew().crew().kickoff()
        import pdb;pdb.set_trace()
        print("README已生成")

    @listen(run_readme_merge)
    def github_pr(self):
        print("\n提交github pr")
        import pdb;pdb.set_trace()
        GithubPRCrew().crew().kickoff()
        import pdb;pdb.set_trace()
        print("已提交PR")

    @listen(github_pr)
    def wait_next_run(self):
        print("\n 流程结束,等待60秒可重新运行...")
        time.sleep(60)

def kickoff():
    MODEL_LIST=["Llama-3.1-8B-Instruct","Llama-3.1-8B-Instruct-FP8"]
    REMOTE_FOLDER="data/remote_folder"
    SSH_CONFIG={
        "hostname":"127.0.0.1",
        "port":22,
        "user_name":"yuchangrui",
        "password":"Ycr2wy1314"
    }
    GITHUB_CONFIG={
        "token":"",
        "repo_owner":"",
        "repo_name":""
    }
    flow=ModelWorkflowFlow(
        model_list=MODEL_LIST,
        remote_folder=REMOTE_FOLDER,
        ssh_config=SSH_CONFIG,
        github_config=GITHUB_CONFIG
    )
    flow.kickoff()

if __name__=="__main__":
    kickoff()

        