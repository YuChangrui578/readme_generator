from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from crewai import Agent,Task,Crew
from readme_generator.tools.memory_tool import GlobalMemory
from readme_generator.crews.input_parser_crew import InputParserCrew
from readme_generator.crews.github_pr_crew import GithubPRCrew
from readme_generator.crews.model_search_crew import ModelSearchCrew
from readme_generator.crews.readme_generate_crew import ReadmeGeneratorCrew
from readme_generator.crews.remote_execution_crew import RemoteExecutionCrew

app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class State:
    def __init__(self):
        self.step=0
        self.finished=False 

state=State()
input_parser=InputParserCrew()
model_searcher=ModelSearchCrew()
readme_generator=ReadmeGeneratorCrew()
remote_executor=RemoteExecutionCrew()
github_pr=GithubPRCrew()

Agent_list=[model_searcher,readme_generator,remote_executor,github_pr]
Agent_name=[
    "ModelSearchCrew",
    "ReadmeGeneratorCrew",
    "RemoteExecutionCrew",
    "GithubPRCrew",
]

class UsrRequest(BaseModel):
    user_input:str

def run_current_agent():
    if state.step>=len(Agent_list):
        state.finished=True
        return {
            "agent":"Completed",
            "final_output":"All agents finished.",
            "finished":True
        }
    agent=Agent_list[state.step]
    output=agent.crew().kickoff()
    final_out=output.final_output if hasattr(output,"final_output") else str(output)
    return {
        "agent":Agent_name[state.step],
        "final_output":final_out,
        "finished":False
    }

@app.post("/api/start")
def start(req:UsrRequest):
    state.step=0
    state.finished=False
    input_parser.crew().kickoff()
    return run_current_agent()

@app.post("/api/next")
def next_agent():
    state.step+=1
    return run_current_agent()

if __name__=="__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,        # 开发模式：文件修改自动重启
        log_level="debug",  # 调试日志
        reload_dirs=["./"]  # 监听当前目录所有文件
    )
