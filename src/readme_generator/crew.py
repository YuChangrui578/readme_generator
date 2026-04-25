from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from .crews.github_pr_crew import GithubPRCrew
from .crews.input_parser_crew import InputParserCrew
from .crews.model_search_crew import ModelSearchCrew
from .crews.readme_generate_crew import ReadmeGeneratorCrew
from .crews.remote_execution_crew import RemoteExecutionCrew
from .tools.memory_tool import GlobalMemory


DEFAULT_REFERENCE_FOLDER = Path(__file__).resolve().parent / "reference_example"
DEFAULT_STAGE_ORDER = [
    "input_parser",
    "model_search",
    "readme_generation",
    "remote_execution",
    "github_pr",
]


def load_reference_files(folder_path: str, recursive: bool = False) -> Dict[str, str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    md_pattern = "**/*.md" if recursive else "*.md"
    js_pattern = "**/*.js" if recursive else "*.js"
    md_files = sorted([p for p in folder.glob(md_pattern) if p.suffix.lower() in [".md", ".markdown"]], key=lambda p: str(p))
    js_files = sorted([p for p in folder.glob(js_pattern) if p.suffix.lower() == ".js"], key=lambda p: str(p))
    md_path = next(
        (p for p in md_files if p.name.lower() in {"readme.md", "reference.md"}),
        md_files[0] if md_files else None,
    )
    js_path = next((p for p in js_files if p.name == "index.js"), js_files[0] if js_files else None)
    return {
        "ref_md": md_path.read_text(encoding="utf-8") if md_path else "",
        "ref_index_js": js_path.read_text(encoding="utf-8") if js_path else "",
    }


class WorkflowInput(BaseModel):
    input_text: str = ""
    model_list: List[str] = Field(default_factory=list)
    github_url: List[str] = Field(default_factory=list)
    skip_stages: List[str] = Field(default_factory=list)
    remote_folder: str = ""
    ssh_config: Dict[str, Any] = Field(default_factory=dict)
    github_config: Dict[str, Any] = Field(default_factory=dict)
    ref_md: str = ""
    ref_index_js: str = ""
    reference_folder: str = str(DEFAULT_REFERENCE_FOLDER)


class WorkflowState(BaseModel):
    id: str = "readme-workflow-default"
    stage_results: List[Dict[str, Any]] = Field(default_factory=list)


def build_legacy_workflow_input() -> WorkflowInput:
    """Legacy preset data from the original runnable script."""
    other_github={
        "github_config": {
            "repo_owner": "YuChangrui578",
            "repo_name": "readme_example",
            "base_branch": "main",
            "head_branch": "dev",
            "pr_title": "update docs bundle",
            "pr_description": "multi-file publish",
            "commit_message": "docs: batch publish",
            "publish_items": [
                { "path": "Xeon/Llama/README.md", "content_key": "family_md" },
                { "path": "Xeon/Llama/index.js", "content_key": "family_index_js" },
                { "path": "Xeon/Common/extra.txt", "content": "custom text content" }
            ]
        }
    }
    return WorkflowInput(
        input_text=(
            '{"model_list": ["Llama-3.2-3B-quantized.w8a8", "Llama-3.2-3B-Instruct-FP8", '
            '"Llama-3.2-3B-Instruct-AWQ"], '
            '"github_url": ["", "", "https://github.com/jianan-gu/sglang/tree/cpu_optimized"]}'
        ),
        model_list=[
            "Llama-3.2-3B-quantized.w8a8",
            "Llama-3.2-3B-Instruct-FP8",
            "Llama-3.2-3B-Instruct-AWQ",
        ],
        github_url=[
            "",
            "",
            "https://github.com/jianan-gu/sglang/tree/cpu_optimized",
        ],
        skip_stages=[],
        remote_folder="/home/sdp/changrui",
        ssh_config={
            "hostname": "10.165.58.104",
            "port": 22,
            "user_name": "sdp",
            "password": "$harktank2Go",
        },
        github_config={
            "github_token": "",
            "repo_owner": "YuChangrui578",
            "repo_name": "readme_example",
            "base_branch": "main",
            "head_branch": "dev",
            "pr_title": "test",
            "pr_description": "test_github_pr",
            "commit_message": "test",
            "path": "Xeon/Llama/",
        },
        reference_folder=str(DEFAULT_REFERENCE_FOLDER),
    )


def build_github_only_legacy_workflow_input() -> WorkflowInput:
    """Legacy preset data for testing GitHub stage only."""
    return build_legacy_workflow_input()


class ReadmeWorkflowCrew(Flow[WorkflowState]):
    """CrewAI Flow orchestration using @start/@listen."""

    initial_state = WorkflowState
    _crew_map: Dict[str, Type] = {
        "input_parser": InputParserCrew,
        "model_search": ModelSearchCrew,
        "readme_generation": ReadmeGeneratorCrew,
        "remote_execution": RemoteExecutionCrew,
        "github_pr": GithubPRCrew,
    }

    def __init__(
        self,
        workflow_input: WorkflowInput,
        enabled_stages: Optional[List[str]] = None,
        memory: Optional[GlobalMemory] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.workflow_input = workflow_input
        self.global_memory = memory or GlobalMemory()
        self.enabled_stages = enabled_stages or [
            stage for stage in DEFAULT_STAGE_ORDER if stage not in workflow_input.skip_stages
        ]
        self._validate_stages()
        self._prepare_memory()

    def _validate_stages(self) -> None:
        unknown = [name for name in self.enabled_stages if name not in self._crew_map]
        if unknown:
            raise ValueError(f"Unsupported stages: {unknown}")

    def _prepare_memory(self) -> None:
        cfg = self.workflow_input

        if not cfg.input_text and (cfg.model_list or cfg.github_url):
            cfg.input_text = (
                "WorkflowInput JSON:\n"
                + str(
                    {
                        "model_list": cfg.model_list or [],
                        "github_url": cfg.github_url or [],
                    }
                )
            )

        ref_md = cfg.ref_md
        ref_index_js = cfg.ref_index_js
        if cfg.reference_folder and (not ref_md or not ref_index_js):
            refs = load_reference_files(cfg.reference_folder)
            ref_md = ref_md or refs.get("ref_md", "")
            ref_index_js = ref_index_js or refs.get("ref_index_js", "")

        if cfg.input_text:
            self.global_memory.memory_store("input_text", cfg.input_text)
        if cfg.model_list:
            self.global_memory.memory_store("model_list", cfg.model_list)

        self.global_memory.memory_store("github_url", cfg.github_url)
        self.global_memory.memory_store("remote_folder", cfg.remote_folder)
        self.global_memory.memory_store("ssh_config", cfg.ssh_config)
        self.global_memory.memory_store("github_config", cfg.github_config)
        self.global_memory.memory_store("ref_md", ref_md)
        self.global_memory.memory_store("ref_index_js", ref_index_js)
        # family_* must be produced by generation stage, not prefilled from reference.
        self.global_memory.memory_store("family_md", "")
        self.global_memory.memory_store("family_index_js", "")
        self.global_memory.memory_store("family_content", "")
        self.global_memory.save_to_file()

    def _is_enabled(self, stage_name: str) -> bool:
        return stage_name in self.enabled_stages

    def _run_stage(self, stage_name: str) -> Dict[str, Any]:
        print(f"\n=== Running stage: {stage_name} ===")
        crew_cls = self._crew_map[stage_name]
        try:
            crew_instance = crew_cls(global_memory=self.global_memory)
        except TypeError as e:
            # Keep compatibility with crews that still use no-arg constructors.
            if "unexpected keyword argument 'global_memory'" in str(e) or "takes no arguments" in str(e):
                crew_instance = crew_cls()
            else:
                raise
        output = crew_instance.crew().kickoff()
        final_output = self._consume_stage_output(output)
        print(f"=== Finished stage: {stage_name} ===")
        return {"stage": stage_name, "final_output": final_output, "skipped": False}

    @staticmethod
    def _normalize_stream_text(text: str) -> str:
        raw = str(text or "")
        if not raw:
            return raw
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 6:
            return raw.strip()
        short_lines = [ln for ln in lines if len(ln) <= 20]
        # If stream is fragmented into many short lines (token/word-level), join them.
        if len(short_lines) >= int(len(lines) * 0.75):
            joined = " ".join(lines)
            joined = " ".join(joined.split())
            return joined.strip()
        return raw.strip()

    def _consume_stage_output(self, output: Any) -> str:
        if isinstance(output, (str, bytes)):
            text = output.decode() if isinstance(output, bytes) else output
            return self._normalize_stream_text(text)

        if hasattr(output, "final_output"):
            return str(output.final_output)

        if isinstance(output, Iterable):
            text_collected: List[str] = []
            event_collected: List[str] = []
            last_agent = "agent"
            for chunk in output:
                chunk_type = getattr(chunk, "chunk_type", None)
                if chunk_type == "text":
                    agent = getattr(getattr(chunk, "agent", None), "role", "agent")
                    last_agent = agent
                    content = getattr(chunk, "content", "")
                    if content:
                        text_collected.append(str(content))
                elif chunk_type == "tool_use":
                    tool_name = getattr(chunk, "tool_name", "tool")
                    tool_input = getattr(chunk, "tool_input", "")
                    print(f"\n[tool_use] {tool_name}: {tool_input}")
                    event_collected.append(f"[tool_use] {tool_name}: {tool_input}")
                else:
                    text = str(chunk)
                    print(text)
                    event_collected.append(text)
            merged_text = self._normalize_stream_text("".join(text_collected).strip())
            if merged_text:
                print(f"[{last_agent}] {merged_text}")
            print()
            if event_collected:
                merged_events = "\n".join(event_collected).strip()
                if merged_text:
                    return f"{merged_text}\n{merged_events}".strip()
                return merged_events
            return merged_text

        return str(output)

    def _run_or_skip(self, stage_name: str) -> Dict[str, Any]:
        if not self._is_enabled(stage_name):
            print(f"\n=== Skipping stage: {stage_name} ===")
            print(f"Skip method: add '{stage_name}' to WorkflowInput.skip_stages or exclude it from enabled_stages.")
            result = {
                "stage": stage_name,
                "skipped": True,
                "skip_method": f"WorkflowInput.skip_stages += ['{stage_name}']",
            }
        else:
            result = self._run_stage(stage_name)
        self.state.stage_results.append(result)
        return result

    @start()
    def run_input_parser(self):
        return self._run_or_skip("input_parser")

    @listen(run_input_parser)
    def run_model_search(self):
        return self._run_or_skip("model_search")

    @listen(run_model_search)
    def run_readme_generation(self):
        return self._run_or_skip("readme_generation")

    @listen(run_readme_generation)
    def run_remote_execution(self):
        return self._run_or_skip("remote_execution")

    @listen(run_remote_execution)
    def run_github_pr(self):
        return self._run_or_skip("github_pr")

    def run(self) -> List[Dict[str, Any]]:
        self.state.stage_results = []
        self.kickoff()
        return self.state.stage_results


def kickoff(
    workflow_input: Optional[WorkflowInput] = None,
    enabled_stages: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    workflow = ReadmeWorkflowCrew(
        workflow_input=workflow_input or WorkflowInput(),
        enabled_stages=enabled_stages,
    )
    return workflow.run()
