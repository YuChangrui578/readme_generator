from __future__ import annotations

import json
import os
import time
from base64 import b64decode
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from .crews.github_pr_crew import GithubPRCrew
from .crews.input_parser_crew import InputParserCrew
from .crews.model_search_crew import ModelSearchCrew
from .crews.post_remote_refine_crew import PostRemoteRefineCrew
from .crews.readme_generate_crew import ReadmeGeneratorCrew
from .crews.remote_execution_crew import RemoteExecutionCrew
from .tools.input_parser_tool import InternelParserLLM
from .tools.generate_readme_tool import GenerateReadmeTool
from .tools.remote_exec_tool import RemoteExecutionTool
from .tools.memory_tool import GlobalMemory, resolve_memory_path


DEFAULT_REFERENCE_FOLDER = Path(__file__).resolve().parent / "reference_example"
DEFAULT_STAGE_ORDER = [
    "input_parser",
    "model_search",
    "readme_generation",
    "remote_execution",
    "post_remote_refine",
    "github_pr",
]


def load_reference_files(folder_path: str, recursive: bool = False) -> Dict[str, str]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    md_pattern = "**/*.mdx" if recursive else "*.mdx"
    js_pattern = "**/*.jsx" if recursive else "*.jsx"
    md_files = sorted([p for p in folder.glob(md_pattern) if p.suffix.lower() in [".mdx"]], key=lambda p: str(p))
    js_files = sorted([p for p in folder.glob(js_pattern) if p.suffix.lower() == ".jsx"], key=lambda p: str(p))
    md_path = next(
        (p for p in md_files if p.name.lower() in {"readme.mdx", "reference.mdx"}),
        md_files[0] if md_files else None,
    )
    js_path = next((p for p in js_files if p.name == "index.jsx"), js_files[0] if js_files else None)
    return {
        "ref_md": md_path.read_text(encoding="utf-8") if md_path else "",
        "ref_index_js": js_path.read_text(encoding="utf-8") if js_path else "",
    }


def _parse_github_source_url(url: str) -> Dict[str, str]:
    parsed = urlparse(url or "")
    host = (parsed.netloc or "").lower()
    if host not in {"github.com", "www.github.com"}:
        raise ValueError(f"Unsupported GitHub URL host: {host}")

    parts = [p for p in (parsed.path or "").strip("/").split("/") if p]
    if len(parts) < 4:
        raise ValueError(f"Invalid GitHub URL path: {parsed.path}")
    owner, repo, kind, branch = parts[0], parts[1], parts[2], parts[3]
    if kind not in {"tree", "blob"}:
        raise ValueError(f"Unsupported GitHub URL kind '{kind}', expected tree/blob")
    sub_path = "/".join(parts[4:]).strip("/")
    return {"owner": owner, "repo": repo, "branch": branch, "path": sub_path, "kind": kind}


def _urlopen_with_retry(req: Request, timeout: int = 30, retries: int = 2, backoff_sec: float = 1.2):
    last_exc: Exception | None = None
    for idx in range(retries + 1):
        try:
            return urlopen(req, timeout=timeout)
        except HTTPError as e:
            # 4xx (except 429) should fail fast; 5xx and 429 can retry.
            if 400 <= e.code < 500 and e.code != 429:
                raise
            last_exc = e
        except URLError as e:
            last_exc = e
        except TimeoutError as e:
            last_exc = e
        if idx < retries:
            time.sleep(backoff_sec * (idx + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("urlopen retry failed without exception")


def _proxy_url() -> str:
    return (
        os.getenv("https_proxy", "").strip()
        or os.getenv("HTTPS_PROXY", "").strip()
        or os.getenv("http_proxy", "").strip()
        or os.getenv("HTTP_PROXY", "").strip()
        or "http://proxy-dmz.intel.com:912"
    )


def _urlopen_with_proxy_retry(req: Request, timeout: int = 30, retries: int = 2, backoff_sec: float = 1.2):
    proxy = _proxy_url()
    opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
    last_exc: Exception | None = None
    for idx in range(retries + 1):
        try:
            return opener.open(req, timeout=timeout)
        except HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                raise
            last_exc = e
        except URLError as e:
            last_exc = e
        except TimeoutError as e:
            last_exc = e
        if idx < retries:
            time.sleep(backoff_sec * (idx + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("urlopen proxy retry failed without exception")


def _open_external_url(req: Request, timeout: int = 30, retries: int = 2):
    # External sites (e.g., GitHub) should use proxy in this environment.
    try:
        return _urlopen_with_proxy_retry(req, timeout=timeout, retries=retries)
    except Exception:
        # Keep a direct fallback for environments where proxy is unavailable.
        return _urlopen_with_retry(req, timeout=timeout, retries=max(1, retries - 1))


def _github_get_json(api_url: str, token: str = "") -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "readme-generator",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(api_url, headers=headers, method="GET")
    with _open_external_url(req, timeout=30, retries=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _github_get_text(download_url: str, token: str = "") -> str:
    headers = {"User-Agent": "readme-generator"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(download_url, headers=headers, method="GET")
    with _open_external_url(req, timeout=30, retries=2) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _fetch_url_text(url: str, token: str = "") -> str:
    headers = {"User-Agent": "readme-generator"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers, method="GET")
    with _open_external_url(req, timeout=30, retries=2) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _collect_github_files(
    owner: str,
    repo: str,
    branch: str,
    folder_path: str,
    token: str,
    suffixes: tuple[str, ...],
) -> List[Dict[str, str]]:
    rel_path = (folder_path or "").strip("/")
    base = f"https://api.github.com/repos/{owner}/{repo}/contents"
    api_url = f"{base}/{rel_path}?ref={branch}" if rel_path else f"{base}?ref={branch}"
    payload = _github_get_json(api_url, token=token)
    items = payload if isinstance(payload, list) else [payload]

    files: List[Dict[str, str]] = []
    for item in items:
        item_type = item.get("type")
        if item_type == "dir":
            files.extend(
                _collect_github_files(
                    owner=owner,
                    repo=repo,
                    branch=branch,
                    folder_path=item.get("path", ""),
                    token=token,
                    suffixes=suffixes,
                )
            )
            continue
        if item_type != "file":
            continue
        path = item.get("path", "")
        lower_path = path.lower()
        if not lower_path.endswith(suffixes):
            continue

        content = ""
        if item.get("download_url"):
            content = _github_get_text(item["download_url"], token=token)
        elif item.get("content"):
            content = b64decode(item.get("content", "")).decode("utf-8", errors="ignore")
        files.append({"path": path, "content": content})
    return files


def _bundle_files(files: List[Dict[str, str]]) -> str:
    sections: List[str] = []
    for item in sorted(files, key=lambda x: x.get("path", "")):
        path = item.get("path", "")
        content = (item.get("content", "") or "").strip()
        if not content:
            continue
        sections.append(f"<!-- source: {path} -->\n{content}")
    return "\n\n".join(sections).strip()


def _pick_primary_markdown(files: List[Dict[str, str]]) -> str:
    if not files:
        return ""
    sorted_files = sorted(files, key=lambda x: x.get("path", "").lower())
    for item in sorted_files:
        name = item.get("path", "").split("/")[-1].lower()
        if name in {"readme.md", "reference.md"}:
            return item.get("content", "") or ""
    return sorted_files[0].get("content", "") or ""


def _pick_primary_index_js(files: List[Dict[str, str]]) -> str:
    if not files:
        return ""
    sorted_files = sorted(files, key=lambda x: x.get("path", "").lower())
    for item in sorted_files:
        if item.get("path", "").split("/")[-1] == "index.js":
            return item.get("content", "") or ""
    return sorted_files[0].get("content", "") or ""


def _load_files_from_source_url(url: str, suffixes: tuple[str, ...], github_token: str = "") -> List[Dict[str, str]]:
    if not url:
        return []
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host in {"github.com", "www.github.com"}:
        info = _parse_github_source_url(url)
        path = info["path"]
        kind = info.get("kind", "")
        # "tree" means directory; "blob" may be file or folder-like link.
        try:
            return _collect_github_files(
                owner=info["owner"],
                repo=info["repo"],
                branch=info["branch"],
                folder_path=path,
                token=github_token,
                suffixes=suffixes,
            )
        except Exception:
            # Fallback 1: for blob-like file path, try raw GitHub directly.
            raw_url = f"https://raw.githubusercontent.com/{info['owner']}/{info['repo']}/{info['branch']}/{path}"
            if path.lower().endswith(suffixes):
                try:
                    return [{"path": path, "content": _fetch_url_text(raw_url, token=github_token)}]
                except Exception:
                    try:
                        return [{"path": path, "content": _fetch_url_text(url, token=github_token)}]
                    except Exception:
                        print(f"[WARN] failed to load source file from URL: {url}")
                        return []
            # Fallback 2: for folder/tree URL, at least keep page content as source context.
            if kind == "tree":
                try:
                    html = _fetch_url_text(url, token=github_token)
                    pseudo_name = (path.split("/")[-1] if path else "source") + (suffixes[0] if suffixes else ".txt")
                    return [{"path": pseudo_name, "content": html}]
                except Exception:
                    print(f"[WARN] failed to load source folder page content: {url}")
                    return []
            print(f"[WARN] failed to load GitHub source URL: {url}")
            return []
    # Generic web source fallback: fetch one page/document as single source entry.
    text = _fetch_url_text(url)
    parsed_path = (parsed.path or "").strip("/")
    name = parsed_path.split("/")[-1] if parsed_path else "source.txt"
    if not name.lower().endswith(suffixes):
        for ext in suffixes:
            if ext.startswith("."):
                name += ext
                break
    return [{"path": name, "content": text}]


import re as _re

def _resolve_js_url_from_mdx(md_url: str, github_token: str = "") -> str:
    """Given a GitHub URL for a .md/.mdx file, fetch its content and look for a component
    import statement (e.g. ``import Foo from '@site/src/components/...'``).
    Returns the corresponding GitHub blob URL for the component's index.js/index.jsx, or "".

    Supports Docusaurus-style @site imports and relative imports.
    """
    if not md_url:
        return ""
    try:
        info = _parse_github_source_url(md_url)
        owner, repo, branch, md_path = info["owner"], info["repo"], info["branch"], info["path"]
    except Exception:
        return ""

    # Fetch the raw markdown text
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{md_path}"
    try:
        text = _fetch_url_text(raw_url, token=github_token)
    except Exception:
        try:
            text = _fetch_url_text(md_url, token=github_token)
        except Exception:
            return ""

    # Look for component import in MDX — several patterns supported
    # Pattern 1: absolute repo-root path import (sgl-cookbook convention)
    #   e.g.  import { Gemma4Deployment } from '/src/snippets/autoregressive/gemma4-deployment.jsx';
    # In Docusaurus, /src/ is relative to the project root = first dir component of MDX path.
    abs_import = _re.search(
        r"import\s+[^'\"]+from\s+['\"]/(src/[^'\"]+\.jsx?)['\"]",
        text,
    )
    if abs_import:
        comp_path_rel = abs_import.group(1)  # e.g. src/snippets/autoregressive/gemma4-deployment.jsx
        md_parts = [p for p in md_path.split("/") if p]
        docs_root = md_parts[0] if len(md_parts) >= 2 else ""
        comp_path = f"{docs_root}/{comp_path_rel}" if docs_root else comp_path_rel
        return f"https://github.com/{owner}/{repo}/blob/{branch}/{comp_path}"

    # Pattern 2: Docusaurus @site import
    #   e.g.  import Llama31Config from '@site/src/components/Llama31ConfigGenerator';
    site_import = _re.search(
        r"import\s+\w+\s+from\s+['\"]@site/([^'\"]+)['\"]",
        text,
    )
    if site_import:
        comp_rel = site_import.group(1).strip("/")
        # Prefer index.jsx, fallback to index.js
        for ext in ("index.jsx", "index.js"):
            candidate_path = f"{comp_rel}/{ext}" if not comp_rel.endswith((".js", ".jsx")) else comp_rel
            js_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{candidate_path}"
            return js_url

    # Pattern 3: relative import like ../components/FooGenerator or ../../base/ConfigGenerator
    rel_import = _re.search(
        r"import\s+\w+\s+from\s+['\"](\.\.[^'\"]+)['\"]",
        text,
    )
    if rel_import:
        rel = rel_import.group(1).strip()
        md_dir = "/".join(md_path.split("/")[:-1])
        # Resolve relative path manually
        parts = (md_dir + "/" + rel).split("/")
        resolved: List[str] = []
        for p in parts:
            if p == "..":
                if resolved:
                    resolved.pop()
            elif p not in ("", "."):
                resolved.append(p)
        comp_path = "/".join(resolved)
        for ext in ("index.jsx", "index.js"):
            candidate = f"{comp_path}/{ext}" if not comp_path.endswith((".js", ".jsx")) else comp_path
            return f"https://github.com/{owner}/{repo}/blob/{branch}/{candidate}"

    # Pattern 3: look for JSX component usage <!-- 组件引用：ComponentName --> or <ComponentName />
    # and guess the path from component name convention
    comp_name_m = _re.search(r"<!--\s*组件引用[：:]\s*(\w+)\s*-->|<(\w+ConfigGenerator)\s*/?>", text)
    if comp_name_m:
        comp_name = (comp_name_m.group(1) or comp_name_m.group(2) or "").strip()
        if comp_name:
            # Standard sgl-cookbook convention: src/components/autoregressive/<Name>/index.js
            comp_path = f"src/components/autoregressive/{comp_name}/index.js"
            return f"https://github.com/{owner}/{repo}/blob/{branch}/{comp_path}"

    return ""


def load_reference_files_from_github_folders(
    md_folder_url: str,
    js_folder_url: str,
    github_token: str = "",
) -> Dict[str, str]:
    try:
        md_files = _load_files_from_source_url(md_folder_url, suffixes=(".md", ".mdx", ".markdown"), github_token=github_token)
    except Exception:
        print(f"[WARN] md source fetch failed: {md_folder_url}")
        md_files = []
    try:
        js_files = _load_files_from_source_url(js_folder_url, suffixes=(".js", ".jsx"), github_token=github_token)
    except Exception:
        print(f"[WARN] js source fetch failed: {js_folder_url}")
        js_files = []
    return {
        "ref_md": _bundle_files(md_files) or _pick_primary_markdown(md_files),
        "ref_index_js": _pick_primary_index_js(js_files),
        "source_md_files": md_files,
        "source_js_files": js_files,
    }


class WorkflowInput(BaseModel):
    memory_profile: str = ""
    generation_mode: str = "reference"
    github_md_folder_url: str = ""
    github_js_folder_url: str = ""
    source_md_url: str = ""
    source_js_url: str = ""
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
        generation_mode="reference",
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
        remote_folder="/home/changrui",
        ssh_config={
            "hostname": "10.112.229.29",
            "port": 22,
            "user_name": "root",
            "password": "intel,123",
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


def build_source_url_workflow_input() -> WorkflowInput:
    """URL-source preset data for Llama 3.1 models."""
    return WorkflowInput(
        memory_profile="url_source",
        generation_mode="web_sources",
        source_md_url="https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Llama/Llama3.1.mdx",
        input_text=(
            "请基于下面两个来源做 Llama3.1 的文档增强：README 来源是 "
            "https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Llama/Llama3.1.mdx ，"
            "请先从这些文件内容里推断模型列表，再继续后续流程。"
        ),
        model_list=[
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B-FP8",
            "meta-llama/Llama-3.1-8B-Instruct",
            "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
            "meta-llama/Llama-3.1-8B-Instruct-FP8",
            "meta-llama/Llama-3.1-70B",
            "meta-llama/Llama-3.1-70B-FP8",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct-FP8",
            "meta-llama/Llama-3.1-405B",
            "meta-llama/Llama-3.1-405B-FP8",
            "meta-llama/Llama-3.1-405B-Instruct",
            "meta-llama/Llama-3.1-405B-Instruct-FP8",
        ],
        github_url=[],
        skip_stages=[],
        remote_folder="",
        ssh_config={},
        github_config={},
        reference_folder=str(DEFAULT_REFERENCE_FOLDER),
    )


class ReadmeWorkflowCrew(Flow[WorkflowState]):
    """CrewAI Flow orchestration using @start/@listen."""

    initial_state = WorkflowState
    _crew_map: Dict[str, Type] = {
        "input_parser": InputParserCrew,
        "model_search": ModelSearchCrew,
        "readme_generation": ReadmeGeneratorCrew,
        "remote_execution": RemoteExecutionCrew,
        "post_remote_refine": PostRemoteRefineCrew,
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
        default_profile = "url_source" if workflow_input.generation_mode in {"web_sources", "github_folders"} else "legacy"
        profile = (workflow_input.memory_profile or "").strip() or default_profile
        self.global_memory = memory or GlobalMemory(persist_path=resolve_memory_path(profile))
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

        if cfg.input_text:
            self.global_memory.memory_store("input_text", cfg.input_text)

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

        # ── Auto-fill from input_text (only when WorkflowInput fields are empty) ──────
        # Run a cheap deterministic extraction pass so downstream stages (and the
        # input_parser agent) start with the correct context even when the caller
        # supplied only a free-form input_text.
        github_token_early = ""
        if isinstance(cfg.github_config, dict):
            github_token_early = str(cfg.github_config.get("github_token") or "").strip()

        if cfg.input_text:
            # Step A: extract source_md_url / model_list from input_text if not already set.
            _parsed_input: Dict[str, Any] = InternelParserLLM._extract_from_workflow_payload(cfg.input_text)
            if not _parsed_input:
                _parsed_input = InternelParserLLM._fallback_parse(cfg.input_text)

            if isinstance(_parsed_input, dict):
                # source_md_url: fill only if not already set
                if not cfg.source_md_url:
                    cfg.source_md_url = str(_parsed_input.get("source_md_url") or "").strip()
                # source_js_url: fill only if not already set
                if not cfg.source_js_url:
                    cfg.source_js_url = str(_parsed_input.get("source_js_url") or "").strip()
                # model_list: fill only if not already set
                if not cfg.model_list:
                    _ml = _parsed_input.get("model_list")
                    if isinstance(_ml, list) and _ml:
                        cfg.model_list = _ml
                # generation_mode: elevate to web_sources if URLs found
                if cfg.source_md_url and not cfg.generation_mode:
                    cfg.generation_mode = str(_parsed_input.get("generation_mode") or "web_sources").strip().lower()

        # ──────────────────────────────────────────────────────────────────────────────
        # NOTE: source_js_url derivation from MDX content is intentionally deferred to
        # the readme_generation stage, where the MDX has already been fetched into memory.
        # See GenerateReadmeTool._ensure_source_js_from_ref_md() for that logic.

        ref_md = cfg.ref_md
        ref_index_js = cfg.ref_index_js
        source_md_files: List[Dict[str, str]] = []
        source_js_files: List[Dict[str, str]] = []
        mode = (cfg.generation_mode or "reference").strip().lower()
        if mode == "reference" and (
            cfg.source_md_url or cfg.source_js_url or cfg.github_md_folder_url or cfg.github_js_folder_url
        ):
            mode = "web_sources"
            cfg.generation_mode = mode
        if mode == "reference" and cfg.input_text and not (cfg.model_list or cfg.github_url):
            hinted = InternelParserLLM._extract_from_workflow_payload(cfg.input_text)
            if not hinted:
                hinted = InternelParserLLM._fallback_parse(cfg.input_text)
            hinted_md = str(hinted.get("source_md_url") or "").strip() if isinstance(hinted, dict) else ""
            hinted_js = str(hinted.get("source_js_url") or "").strip() if isinstance(hinted, dict) else ""
            if hinted_md and hinted_js:
                mode = str(hinted.get("generation_mode") or "web_sources").strip().lower()
                cfg.generation_mode = mode
                cfg.source_md_url = cfg.source_md_url or hinted_md
                cfg.source_js_url = cfg.source_js_url or hinted_js

        github_token = ""
        if isinstance(cfg.github_config, dict):
            github_token = str(cfg.github_config.get("github_token") or "").strip()
        if mode in {"github_folders", "web_sources"}:
            md_url = cfg.github_md_folder_url or cfg.source_md_url
            js_url = cfg.github_js_folder_url or cfg.source_js_url
            if not md_url:
                raise ValueError(
                    "generation_mode=github_folders/web_sources requires at least a md source URL."
                )
            # js_url may be empty here — it will be derived from the fetched MDX content
            # later in the readme_generation stage (GenerateReadmeTool._ensure_source_js_from_ref_md).
            refs = load_reference_files_from_github_folders(
                md_folder_url=md_url,
                js_folder_url=js_url,
                github_token=github_token,
            )
            ref_md = ref_md or refs.get("ref_md", "")
            ref_index_js = ref_index_js or refs.get("ref_index_js", "")
            source_md_files = refs.get("source_md_files", []) or []
            source_js_files = refs.get("source_js_files", []) or []
        elif cfg.reference_folder and (not ref_md or not ref_index_js):
            refs = load_reference_files(cfg.reference_folder)
            ref_md = ref_md or refs.get("ref_md", "")
            ref_index_js = ref_index_js or refs.get("ref_index_js", "")

        if cfg.model_list:
            self.global_memory.memory_store("model_list", cfg.model_list)

        self.global_memory.memory_store("github_url", cfg.github_url)
        self.global_memory.memory_store("generation_mode", mode)
        self.global_memory.memory_store("github_md_folder_url", cfg.github_md_folder_url)
        self.global_memory.memory_store("github_js_folder_url", cfg.github_js_folder_url)
        self.global_memory.memory_store("source_md_url", cfg.source_md_url)
        self.global_memory.memory_store("source_js_url", cfg.source_js_url)
        self.global_memory.memory_store("remote_folder", cfg.remote_folder)
        self.global_memory.memory_store("ssh_config", cfg.ssh_config)

        # In url_source/web_sources mode, auto-populate github_config.publish_items from
        # the source URLs so the github_pr stage writes the generated files back to the
        # same paths they came from (replacing the originals via PR).
        effective_github_config = dict(cfg.github_config) if isinstance(cfg.github_config, dict) else {}
        if mode in {"web_sources", "github_folders"} and not effective_github_config.get("publish_items"):
            md_url_for_publish = (cfg.source_md_url or cfg.github_md_folder_url or "").strip()
            js_url_for_publish = (cfg.source_js_url or cfg.github_js_folder_url or "").strip()
            publish_items: List[Dict[str, Any]] = []
            src_owner = src_repo = src_branch = ""
            try:
                if md_url_for_publish:
                    md_info = _parse_github_source_url(md_url_for_publish)
                    src_owner = md_info["owner"]
                    src_repo = md_info["repo"]
                    src_branch = md_info["branch"]
                    publish_items.append({"path": md_info["path"], "content_key": "family_md"})
            except Exception:
                pass
            try:
                if js_url_for_publish:
                    js_info = _parse_github_source_url(js_url_for_publish)
                    if not src_owner:
                        src_owner = js_info["owner"]
                        src_repo = js_info["repo"]
                        src_branch = js_info["branch"]
                    publish_items.append({"path": js_info["path"], "content_key": "family_index_js"})
            except Exception:
                pass
            if publish_items and src_owner and src_repo:
                effective_github_config.setdefault("repo_owner", src_owner)
                effective_github_config.setdefault("repo_name", src_repo)
                effective_github_config.setdefault("base_branch", src_branch or "main")
                effective_github_config.setdefault("head_branch", f"intel-xeon-support/{src_branch or 'main'}")
                effective_github_config.setdefault("pr_title", "Add Intel Xeon (CPU) deployment support")
                effective_github_config.setdefault(
                    "pr_description",
                    "Auto-generated: adds Intel CPU (Xeon) deployment sections alongside CUDA/AMD.",
                )
                effective_github_config.setdefault("commit_message", "docs: add Intel Xeon CPU deployment support")
                effective_github_config["publish_items"] = publish_items
                print(
                    f"[_prepare_memory] url_source: auto-derived publish_items from source URLs "
                    f"→ {[p['path'] for p in publish_items]} in {src_owner}/{src_repo}"
                )
            cfg.github_config = effective_github_config

        self.global_memory.memory_store("github_config", effective_github_config)
        self.global_memory.memory_store("ref_md", ref_md)
        self.global_memory.memory_store("ref_index_js", ref_index_js)
        self.global_memory.memory_store("source_md_files", source_md_files)
        self.global_memory.memory_store("source_js_files", source_js_files)
        # family_* must be produced by generation stage, not prefilled from reference.
        self.global_memory.memory_store("family_md", "")
        self.global_memory.memory_store("family_index_js", "")
        self.global_memory.memory_store("family_content", "")
        self.global_memory.memory_store("family_js_files", [])
        self.global_memory.memory_store("execution_result", [])
        self.global_memory.memory_store("executed_command", [])
        self.global_memory.memory_store("fail_reason_list", [])
        self.global_memory.memory_store("review_failure_report", "")
        self.global_memory.memory_store("pr_info", {})

        # Build remote payload for remote_executor stage. Distinguish legacy vs url_source.
        try:
            remote_payload: Dict[str, Any] = {}
            gen_mode = mode
            if gen_mode in {"github_folders", "web_sources"}:
                source_md = (cfg.source_md_url or cfg.github_md_folder_url or "").strip()
                source_js = (cfg.source_js_url or cfg.github_js_folder_url or "").strip()
                remote_payload = {
                    "generation_mode": "url_source",
                    "model_list": cfg.model_list or [],
                    "source_urls": {
                        "md": source_md,
                        "js": source_js,
                    },
                    "metadata": {
                        "official_only": True,
                    },
                }
            else:
                # legacy: include content (ref md + index js + input_text), model_list and github_url
                official_models = []
                dev_models = []
                model_to_url = {}
                for idx, m in enumerate(cfg.model_list or []):
                    url = ""
                    try:
                        url = cfg.github_url[idx] if idx < len(cfg.github_url) else ""
                    except Exception:
                        url = ""
                    model_to_url[m] = url
                    if url:
                        dev_models.append(m)
                    else:
                        official_models.append(m)
                remote_payload = {
                    "generation_mode": "legacy",
                    "content": {
                        "family_md": ref_md,
                        "family_index_js": ref_index_js,
                        "input_text": cfg.input_text,
                    },
                    "model_list": cfg.model_list or [],
                    "github_url": cfg.github_url or [],
                    "metadata": {
                        "official": official_models,
                        "dev": dev_models,
                        "model_to_github_url": model_to_url,
                    },
                }
        except Exception:
            fallback_mode = "url_source" if mode in {"web_sources", "github_folders"} else "legacy"
            remote_payload = {"generation_mode": fallback_mode, "model_list": cfg.model_list or []}

        self.global_memory.memory_store("remote_payload", remote_payload)
        self.global_memory.save_to_file()

    def _is_enabled(self, stage_name: str) -> bool:
        return stage_name in self.enabled_stages

    def _run_readme_generation_steps(self) -> Dict[str, Any]:
        """Execute readme_generation as 6 explicit sequential tool steps with per-step output."""
        gt = GenerateReadmeTool
        gt.global_memory = self.global_memory
        gt._pipeline.clear()  # reset transient state for this run
        step_results: Dict[str, Any] = {}

        def _step(n: int, name: str, tool_fn) -> Dict[str, Any]:
            print(f"\n  ┌─ Step {n}/6 [{name}]")
            try:
                result = tool_fn()
                print(f"  └─ {json.dumps(result, ensure_ascii=False, default=str)}")
                return result
            except Exception as e:
                err = {"error": str(e)}
                print(f"  └─ ERROR: {e}")
                return err

        step_results["1_ensure_source_js"] = _step(
            1, "ensure_source_js_url", gt.step_ensure_source_js_url.func
        )
        step_results["2_resolve_mode"] = _step(
            2, "resolve_generation_mode", gt.step_resolve_generation_mode.func
        )
        step_results["3_classify_models"] = _step(
            3, "classify_models", gt.step_classify_models.func
        )
        step_results["4_llm_generate"] = _step(
            4, "run_llm_generation", gt.step_run_llm_generation.func
        )
        step_results["5_postprocess"] = _step(
            5, "postprocess_artifacts", gt.step_postprocess_artifacts.func
        )
        step_results["6_store"] = _step(
            6, "store_final_artifacts", gt.step_store_final_artifacts.func
        )

        all_ok = all("error" not in v for v in step_results.values())
        return {"ok": all_ok, "step_results": step_results}

    def _run_stage(self, stage_name: str) -> Dict[str, Any]:
        print(f"\n=== Running stage: {stage_name} ===")
        if stage_name == "readme_generation":
            # Bypass agent-level LLM for this stage to avoid hard failure/noise when
            # the Crew LLM endpoint is unavailable; run each pipeline step explicitly
            # so intermediate results are visible.
            GenerateReadmeTool.global_memory = self.global_memory
            tool_result = self._run_readme_generation_steps()
            final_output = self._build_readme_generation_output(
                json.dumps(tool_result, ensure_ascii=False)
            )
            self._print_readme_generation_terminal_output()
            self._save_readme_generation_artifacts()
            print(f"=== Finished stage: {stage_name} ===")
            return {"stage": stage_name, "final_output": final_output, "skipped": False}
        if stage_name == "remote_execution":
            # Bypass agent-level LLM for this stage to avoid flow hard-fail when the
            # Crew LLM endpoint is unavailable. Execute remote validations directly.
            RemoteExecutionTool.global_memory = self.global_memory
            direct_result = self._run_remote_execution_direct()
            final_output = json.dumps(direct_result, ensure_ascii=False)
            print(f"=== Finished stage: {stage_name} ===")
            return {"stage": stage_name, "final_output": final_output, "skipped": False}

        crew_cls = self._crew_map[stage_name]
        before_family_md = ""
        before_family_index_js = ""
        if stage_name == "readme_generation":
            before_family_md = str(self.global_memory.memory_retrieve("family_md") or "").strip()
            before_family_index_js = str(self.global_memory.memory_retrieve("family_index_js") or "").strip()
        try:
            crew_instance = crew_cls(global_memory=self.global_memory)
        except TypeError as e:
            # Keep compatibility with crews that still use no-arg constructors.
            if "unexpected keyword argument 'global_memory'" in str(e) or "takes no arguments" in str(e):
                crew_instance = crew_cls()
            else:
                raise
        # Use crew_instance.kickoff() if available so stage-level bypasses (e.g. url_source
        # model_search bypass) are respected; fall back to crew().kickoff() otherwise.
        if hasattr(crew_instance, "kickoff") and callable(crew_instance.kickoff):
            output = crew_instance.kickoff()
        else:
            output = crew_instance.crew().kickoff()
        if stage_name == "readme_generation":
            # Safety net: ensure artifacts in memory are refreshed for current run.
            # If the agent did not actually write artifacts (or left stale values),
            # force local generation from GLOBAL_MEMORY context.
            family_md = str(self.global_memory.memory_retrieve("family_md") or "").strip()
            family_index_js = str(self.global_memory.memory_retrieve("family_index_js") or "").strip()
            unchanged = (
                family_md == before_family_md
                and family_index_js == before_family_index_js
            )
            missing = (not family_md) or (not family_index_js)
            if missing or unchanged:
                GenerateReadmeTool.global_memory = self.global_memory
                try:
                    forced = GenerateReadmeTool.memory_generate_and_store_family_artifacts.func()
                    print(f"[readme_generation][fallback] {forced}")
                except Exception as e:
                    print(f"[readme_generation][fallback][error] {e}")
        final_output = self._consume_stage_output(output)
        if stage_name == "readme_generation":
            final_output = self._build_readme_generation_output(final_output)
        print(f"=== Finished stage: {stage_name} ===")
        return {"stage": stage_name, "final_output": final_output, "skipped": False}

    def _run_remote_execution_direct(self) -> Dict[str, Any]:
        context = RemoteExecutionTool.memory_retrieve_execution_context.func()
        mode = str(context.get("generation_mode") or "").strip().lower()
        stored = 0
        failed = 0

        if mode == "url_source":
            result = RemoteExecutionTool.execute_remote_readme_validation.func("", "")
            fail_reason = None if bool(result.get("ok")) else str(result.get("error") or "remote validation failed")
            if fail_reason:
                failed += 1
            RemoteExecutionTool.memory_store_execution_result.func(
                idx=0,
                command_str="url_source_payload",
                result=result,
                fail_reason=fail_reason,
                updated_readme=None,
            )
            stored += 1
            return {
                "ok": True,
                "mode": "url_source",
                "stored_count": stored,
                "failed_count": failed,
            }

        model_ids = list(context.get("model_id_list") or [])
        model_contents = list(context.get("model_content_list") or [])
        for idx, model_id in enumerate(model_ids):
            model_content = model_contents[idx] if idx < len(model_contents) else ""
            result = RemoteExecutionTool.execute_remote_readme_validation.func(str(model_id), str(model_content))
            fail_reason = None if bool(result.get("ok")) else str(result.get("error") or "remote validation failed")
            if fail_reason:
                failed += 1
            command_value = result.get("request_payload", {}).get("commands") if isinstance(result, dict) else None
            command_str = (
                json.dumps(command_value, ensure_ascii=False)
                if command_value
                else f"validate:{model_id}"
            )
            RemoteExecutionTool.memory_store_execution_result.func(
                idx=idx,
                command_str=command_str,
                result=result,
                fail_reason=fail_reason,
                updated_readme=None,
            )
            stored += 1

        return {
            "ok": True,
            "mode": "legacy",
            "stored_count": stored,
            "failed_count": failed,
        }

    def _build_readme_generation_output(self, fallback_output: str) -> str:
        family_md = str(self.global_memory.memory_retrieve("family_md") or "").strip()
        family_index_js = str(self.global_memory.memory_retrieve("family_index_js") or "").strip()
        if not family_md and not family_index_js:
            return fallback_output
        return json.dumps(
            {
                "family_md": family_md,
                "family_index_js": family_index_js,
            },
            ensure_ascii=False,
        )

    def _print_readme_generation_terminal_output(self) -> None:
        family_md = str(self.global_memory.memory_retrieve("family_md") or "").strip()
        family_index_js = str(self.global_memory.memory_retrieve("family_index_js") or "").strip()
        if not family_md and not family_index_js:
            print("[readme_generation][output] empty family_md/family_index_js")
            return
        print("\nFinal Output:")
        print(
            json.dumps(
                {
                    "family_md": family_md,
                    "family_index_js": family_index_js,
                },
                ensure_ascii=False,
            )
        )

    def _save_readme_generation_artifacts(self) -> None:
        """Write family_md → output.mdx and family_index_js → index.jsx next to this file."""
        family_md = str(self.global_memory.memory_retrieve("family_md") or "").strip()
        family_index_js = str(self.global_memory.memory_retrieve("family_index_js") or "").strip()
        output_dir = Path(__file__).resolve().parent
        if family_md:
            out_md = output_dir / "output.mdx"
            out_md.write_text(family_md, encoding="utf-8")
            print(f"[readme_generation][saved] {out_md}")
        if family_index_js:
            out_js = output_dir / "index.jsx"
            out_js.write_text(family_index_js, encoding="utf-8")
            print(f"[readme_generation][saved] {out_js}")

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
    def run_post_remote_refine(self):
        return self._run_or_skip("post_remote_refine")

    @listen(run_post_remote_refine)
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
