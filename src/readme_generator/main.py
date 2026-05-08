from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

LEGACY_USER_INPUT_TEXT_EXAMPLE = (
    "我现在要处理一组 Llama 3.1 模型，包括 Llama-3.1-8B-Instruct、"
    "Llama-3.1-8B-Instruct-FP8 和 Llama-3.1-8B-Instruct-AWQ。"
    "其中前两个模型走官方 sglang，AWQ 模型要参考这个分支："
    "https://github.com/jianan-gu/sglang/tree/cpu_optimized 。"
    "请按这组模型生成 README 和 index.js。"
)

QWEN3_URL_SOURCE_INPUT_TEXT_EXAMPLE = (
    "请基于下面两个来源做 Qwen3 的文档增强：README 来源是 "
    "https://github.com/sgl-project/sgl-cookbook/blob/main/docs/autoregressive/Qwen/Qwen3.md ，"
    "JS 来源是 "
    "https://github.com/sgl-project/sgl-cookbook/blob/main/src/components/autoregressive/Qwen3ConfigGenerator/ 。"
    "这次只考虑官方 sglang 的安装和启动方式，不要使用 dev branch。"
    "请先从这些文件内容里推断模型列表，再继续后续流程。"
)

def _ensure_no_proxy_for_internal_hosts() -> None:
    hosts = {"10.54.34.78", "10.165.58.104", "127.0.0.1", "localhost"}
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        existing = {item.strip() for item in current.split(",") if item.strip()}
        merged = sorted(existing | hosts)
        os.environ[key] = ",".join(merged)


# Avoid proxy interception for internal LLM / remote API calls.
_ensure_no_proxy_for_internal_hosts()
# Avoid telemetry timeout noise in restricted network environments.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

if __package__ in (None, ""):
    # Support direct execution: python src/readme_generator/main.py
    project_src = Path(__file__).resolve().parent.parent
    if str(project_src) not in sys.path:
        sys.path.append(str(project_src))
    from readme_generator.crew import (
        ReadmeWorkflowCrew as WorkflowRunner,
        WorkflowInput,
        build_github_only_legacy_workflow_input,
        build_legacy_workflow_input,
        build_source_url_workflow_input,
        kickoff as crew_kickoff,
    )
    from readme_generator.tools.input_parser_tool import InternelParserLLM
else:
    from .crew import (
        ReadmeWorkflowCrew as WorkflowRunner,
        WorkflowInput,
        build_github_only_legacy_workflow_input,
        build_legacy_workflow_input,
        build_source_url_workflow_input,
        kickoff as crew_kickoff,
    )
    from .tools.input_parser_tool import InternelParserLLM


def kickoff(
    workflow_input: Optional[WorkflowInput] = None,
    enabled_stages: Optional[List[str]] = None,
):
    prepared_input = prepare_workflow_input(workflow_input or WorkflowInput())
    prepared_stages = prepare_enabled_stages(prepared_input, enabled_stages)
    return crew_kickoff(workflow_input=prepared_input, enabled_stages=prepared_stages)


def _build_input_text_for_mode(workflow_input: WorkflowInput) -> str:
    if (workflow_input.input_text or "").strip():
        return workflow_input.input_text

    mode = (workflow_input.generation_mode or "reference").strip().lower()
    if mode in {"github_folders", "web_sources"}:
        source_md_url = (workflow_input.source_md_url or workflow_input.github_md_folder_url or "").strip()
        source_js_url = (workflow_input.source_js_url or workflow_input.github_js_folder_url or "").strip()
        payload = {
            "generation_mode": mode,
            "source_md_url": source_md_url,
            "source_js_url": source_js_url,
            "official_only": True,
            "model_list": [],
            "github_url": [],
        }
        return json.dumps(payload, ensure_ascii=False)

    if workflow_input.model_list or workflow_input.github_url:
        payload = {
            "model_list": workflow_input.model_list or [],
            "github_url": workflow_input.github_url or [],
        }
        return json.dumps(payload, ensure_ascii=False)
    return ""


def prepare_workflow_input(workflow_input: WorkflowInput) -> WorkflowInput:
    prepared = workflow_input.model_copy(deep=True)
    prepared.input_text = _build_input_text_for_mode(prepared)
    if (prepared.input_text or "").strip():
        inferred = InternelParserLLM._extract_from_workflow_payload(prepared.input_text)
        if not inferred:
            inferred = InternelParserLLM.parse(prepared.input_text)
        inferred_mode = str(inferred.get("generation_mode") or "").strip().lower() if isinstance(inferred, dict) else ""
        explicit_legacy_input = (
            (prepared.generation_mode or "").strip().lower() == "reference"
            and bool(prepared.model_list or prepared.github_url)
            and not any(
                [
                    (prepared.source_md_url or "").strip(),
                    (prepared.source_js_url or "").strip(),
                    (prepared.github_md_folder_url or "").strip(),
                    (prepared.github_js_folder_url or "").strip(),
                ]
            )
        )
        if inferred_mode in {"web_sources", "github_folders", "url_source"}:
            # Keep explicit legacy runs stable: do not auto-flip to url_source when legacy fields are provided.
            if not explicit_legacy_input:
                prepared.generation_mode = "web_sources"
                if not (prepared.source_md_url or "").strip():
                    prepared.source_md_url = str(inferred.get("source_md_url") or "").strip()
                if not (prepared.source_js_url or "").strip():
                    prepared.source_js_url = str(inferred.get("source_js_url") or "").strip()
                prepared.github_url = []
        elif inferred_mode == "reference":
            prepared.generation_mode = "reference"
    if not (prepared.memory_profile or "").strip():
        mode = (prepared.generation_mode or "reference").strip().lower()
        prepared.memory_profile = "url_source" if mode in {"web_sources", "github_folders"} else "legacy"
    return prepared


def prepare_enabled_stages(
    workflow_input: WorkflowInput,
    enabled_stages: Optional[List[str]],
) -> Optional[List[str]]:
    if not enabled_stages:
        return None
    alias_map = {
        "readme_generate": "readme_generation",
        "remote_execute": "remote_execution",
    }
    stages = [alias_map.get(s, s) for s in enabled_stages if s]
    seen = set()
    normalized = []
    for s in stages:
        if s in seen:
            continue
        seen.add(s)
        normalized.append(s)
    return normalized


if __name__ == "__main__":
    print("Running unified workflow default input (legacy).")
    kickoff(
        # workflow_input=build_legacy_workflow_input(),
        workflow_input=build_source_url_workflow_input(),
        # Free-text legacy example (auto-detected as legacy/reference):
        # workflow_input=WorkflowInput(
        #     input_text=LEGACY_USER_INPUT_TEXT_EXAMPLE,
        # ),
        # Free-text source_url example (auto-detected as web_sources/url_source):
        # workflow_input=WorkflowInput(
        #     input_text=QWEN3_URL_SOURCE_INPUT_TEXT_EXAMPLE,
        # ),
        # For stage testing, pass enabled_stages manually, e.g.:
        # enabled_stages=["readme_generate", "remote_execute"],
        enabled_stages=["readme_generate"],
        # For full workflow, leave enabled_stages=None (default).
    )
