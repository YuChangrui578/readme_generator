from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from readme_generator.main import WorkflowInput, WorkflowRunner


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class State:
    def __init__(self) -> None:
        self.runner: Optional[WorkflowRunner] = None
        self.stages: List[str] = []
        self.step: int = 0
        self.finished: bool = False


state = State()


class WorkflowRequest(BaseModel):
    input_text: str = ""
    model_list: List[str] = Field(default_factory=list)
    github_url: List[str] = Field(default_factory=list)
    skip_stages: List[str] = Field(default_factory=list)
    remote_folder: str = ""
    ssh_config: Dict[str, Any] = Field(default_factory=dict)
    github_config: Dict[str, Any] = Field(default_factory=dict)
    ref_md: str = ""
    ref_index_js: str = ""
    reference_folder: Optional[str] = None
    stages: List[str] = Field(default_factory=list)


def _run_current_stage():
    if not state.runner:
        raise HTTPException(status_code=400, detail="Workflow has not started.")
    if state.step >= len(state.stages):
        state.finished = True
        return {"stage": "completed", "finished": True, "result": None}

    stage_name = state.stages[state.step]
    result = state.runner._run_stage(stage_name)
    return {"stage": stage_name, "finished": False, "result": result}


@app.post("/api/start")
def start(req: WorkflowRequest):
    workflow_input = WorkflowInput(
        input_text=req.input_text,
        model_list=req.model_list,
        github_url=req.github_url,
        skip_stages=req.skip_stages,
        remote_folder=req.remote_folder,
        ssh_config=req.ssh_config,
        github_config=req.github_config,
        ref_md=req.ref_md,
        ref_index_js=req.ref_index_js,
        reference_folder=req.reference_folder or WorkflowInput.model_fields["reference_folder"].default,
    )
    state.runner = WorkflowRunner(
        workflow_input=workflow_input,
        enabled_stages=req.stages or None,
    )
    state.stages = state.runner.enabled_stages
    state.step = 0
    state.finished = False
    return _run_current_stage()


@app.post("/api/next")
def next_stage():
    if state.finished:
        return {"stage": "completed", "finished": True, "result": None}
    state.step += 1
    return _run_current_stage()


@app.post("/api/run")
def run_all(req: WorkflowRequest):
    workflow_input = WorkflowInput(
        input_text=req.input_text,
        model_list=req.model_list,
        github_url=req.github_url,
        skip_stages=req.skip_stages,
        remote_folder=req.remote_folder,
        ssh_config=req.ssh_config,
        github_config=req.github_config,
        ref_md=req.ref_md,
        ref_index_js=req.ref_index_js,
        reference_folder=req.reference_folder or WorkflowInput.model_fields["reference_folder"].default,
    )
    runner = WorkflowRunner(workflow_input=workflow_input, enabled_stages=req.stages or None)
    return {"finished": True, "results": runner.run()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        reload_dirs=["./"],
    )
