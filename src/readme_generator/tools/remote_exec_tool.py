import json
import re
from typing import Any, Dict, List, Optional

import requests
from crewai.tools import tool


class RemoteExecutionClient:
    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    @staticmethod
    def extract_commands_from_readme(readme: str) -> List[str]:
        if not readme:
            return []

        commands: List[str] = []
        for block in re.findall(r"```(?:bash|shell)\s*([\s\S]*?)```", readme, flags=re.IGNORECASE):
            for line in block.splitlines():
                cmd = line.strip()
                if not cmd or cmd.startswith("#"):
                    continue
                commands.append(cmd)
        return commands

    @staticmethod
    def _parse_stream_chunks(stream_chunks: List[str]) -> Dict[str, Any]:
        parsed_events: List[Any] = []
        text_events: List[str] = []
        reconstructed_text: List[str] = []
        for chunk in stream_chunks:
            if chunk in ("[DONE]", "DONE"):
                continue
            try:
                parsed = json.loads(chunk)
                parsed_events.append(parsed)
                if isinstance(parsed, dict):
                    choices = parsed.get("choices")
                    if isinstance(choices, list) and choices:
                        delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            reconstructed_text.append(content)
                    text_val = parsed.get("text") or parsed.get("content")
                    if isinstance(text_val, str) and text_val:
                        reconstructed_text.append(text_val)
                continue
            except Exception:
                pass
            text_events.append(chunk)

        merged_text = "".join(reconstructed_text).strip()
        if not merged_text and text_events:
            merged_text = "\n".join(text_events).strip()

        if parsed_events:
            return {"events": parsed_events, "text": merged_text}
        if text_events:
            return {"raw_response": merged_text}
        return {}

    @staticmethod
    def _read_sse_events(response: requests.Response) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        current_event = "message"
        data_lines: List[str] = []

        def flush_event() -> None:
            nonlocal current_event, data_lines
            if not data_lines:
                current_event = "message"
                return
            raw_data = "\n".join(data_lines).strip()
            try:
                parsed_data: Any = json.loads(raw_data)
            except Exception:
                parsed_data = raw_data
            events.append({"event": current_event, "data": parsed_data, "raw_data": raw_data})
            print(f"[remote_stream][{current_event}] {raw_data}")
            current_event = "message"
            data_lines = []

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            chunk = line.strip()
            if not chunk:
                flush_event()
                continue
            if chunk.startswith(":"):
                continue
            if chunk.startswith("event:"):
                current_event = chunk[6:].strip() or "message"
                continue
            if chunk.startswith("data:"):
                data_lines.append(chunk[5:].strip())
                continue
            data_lines.append(chunk)

        flush_event()
        return events

    @staticmethod
    def _parse_sse_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        reconstructed: List[str] = []
        for item in events:
            data = item.get("data")
            if isinstance(data, dict):
                if isinstance(data.get("chunk"), str) and data.get("chunk"):
                    reconstructed.append(data["chunk"])
                    continue
                if isinstance(data.get("content"), str) and data.get("content"):
                    reconstructed.append(data["content"])
                    continue
                if isinstance(data.get("message"), str) and data.get("message"):
                    reconstructed.append(data["message"])
                    continue
            if isinstance(data, str) and data:
                reconstructed.append(data)
        return {"events": events, "text": "".join(reconstructed).strip()}

    def validate_model_readme(
        self,
        request_url: str,
        model_id: str,
        content: str,
        extra_payload: Optional[Dict[str, Any]] = None,
        stream: bool = True,
        include_extracted_commands: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model_id": model_id,
            "content": content,
        }
        if include_extracted_commands:
            payload["commands"] = self.extract_commands_from_readme(content)
        if extra_payload:
            payload.update(extra_payload)

        try:
            if stream:
                response = requests.post(
                    request_url,
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                    headers={"Accept": "text/event-stream"},
                )
                response.raise_for_status()
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if "text/event-stream" in content_type:
                    sse_events = self._read_sse_events(response)
                    parsed_payload = self._parse_sse_events(sse_events)
                    stream_output = [json.dumps(evt, ensure_ascii=False) for evt in sse_events]
                else:
                    stream_chunks: List[str] = []
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        chunk = line.strip()
                        if not chunk:
                            continue
                        stream_chunks.append(chunk)
                        print(f"[remote_stream] {chunk}")
                    parsed_payload = self._parse_stream_chunks(stream_chunks)
                    stream_output = stream_chunks

                return {
                    "ok": True,
                    "status_code": response.status_code,
                    "request_url": request_url,
                    "request_payload": payload,
                    "response": parsed_payload,
                    "stream_output": stream_output,
                    "used_stream": True,
                }

            response = requests.post(
                request_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            try:
                response_payload = response.json()
            except ValueError:
                response_payload = {"raw_response": response.text}
            return {
                "ok": True,
                "status_code": response.status_code,
                "request_url": request_url,
                "request_payload": payload,
                "response": response_payload,
                "used_stream": False,
            }
        except Exception as e:
            return {
                "ok": False,
                "status_code": None,
                "request_url": request_url,
                "response": {},
                "error": str(e),
                "used_stream": stream,
            }


class RemoteExecutionTool:
    client = RemoteExecutionClient()
    global_memory = None

    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) if v is not None else "" for v in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) if v is not None else "" for v in parsed]
            except Exception:
                pass
            return [value]
        return []

    @staticmethod
    def _compose_model_content_from_family(
        model_id: str,
        model_name: str,
        model_url: str,
        github_url: str,
        family_md: str,
        family_index_js: str,
    ) -> str:
        branch_mode = "dev_branch" if str(github_url or "").strip() else "official"
        model_header = (model_id or model_name or "").strip()
        model_url = (model_url or "").strip()
        github_url = (github_url or "").strip()
        family_md = (family_md or "").strip()
        family_index_js = (family_index_js or "").strip()

        if not family_md or not family_index_js:
            return ""

        return (
            f"## Target Model\n"
            f"- model_id: {model_id}\n"
            f"- model_name: {model_name}\n"
            f"- model_url: {model_url}\n"
            f"- mode: {branch_mode}\n"
            f"- github_url: {github_url}\n\n"
            "## Rewriting Requirement\n"
            f"Use the full family README.md and index.js below to derive commands and test logic for ONLY this target model ({model_header}). "
            "Do not execute other model variants.\n\n"
            "## Family README.md (full)\n\n"
            f"{family_md}\n\n"
            "## Family index.js (full)\n\n"
            "```javascript\n"
            f"{family_index_js}\n"
            "```"
        ).strip()

    @staticmethod
    def _resolve_model_content_list() -> Dict[str, Any]:
        model_list = RemoteExecutionTool._normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("model_list") or []
        )
        model_id_list = RemoteExecutionTool._normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("model_id_list") or []
        )
        model_url_list = RemoteExecutionTool._normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("model_url_list") or []
        )
        github_url_list = RemoteExecutionTool._normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("github_url") or []
        )
        family_md = str(
            RemoteExecutionTool.global_memory.memory_retrieve("family_md") or ""
        ).strip()
        family_index_js = str(RemoteExecutionTool.global_memory.memory_retrieve("family_index_js") or "").strip()
        if not family_md:
            raise ValueError("family_md is required for remote execution.")
        if not family_index_js:
            raise ValueError("family_index_js is required for remote execution.")
        if not model_id_list:
            raise ValueError("model_id_list is required for remote execution.")

        derived: List[str] = []
        for i in range(len(model_id_list)):
            derived.append(
                RemoteExecutionTool._compose_model_content_from_family(
                    model_id=model_id_list[i] if i < len(model_id_list) else "",
                    model_name=model_list[i] if i < len(model_list) else "",
                    model_url=model_url_list[i] if i < len(model_url_list) else "",
                    github_url=github_url_list[i] if i < len(github_url_list) else "",
                    family_md=family_md,
                    family_index_js=family_index_js,
                )
            )
        return {
            "model_content_list": derived,
            "warning": "",
        }

    @staticmethod
    def _resolve_request_url() -> str:
        ssh_config = RemoteExecutionTool.global_memory.memory_retrieve("ssh_config") or {}
        # Preferred override: full URL in ssh_config.request_url
        # Otherwise composed by ssh_config.hostname/request_scheme/request_port/request_endpoint
        if ssh_config.get("request_url"):
            return ssh_config["request_url"]

        host = ssh_config.get("hostname")
        if not host:
            raise ValueError("Missing request_url or hostname in ssh_config.")

        scheme = ssh_config.get("request_scheme", "http")
        port = ssh_config.get("request_port", 8000)
        request_stream = bool(ssh_config.get("request_stream", False))
        default_endpoint = "/bkc_test/stream" if request_stream else "/bkc_test"
        endpoint = ssh_config.get("request_endpoint", default_endpoint)
        return f"{scheme}://{host}:{port}{endpoint}"

    @staticmethod
    def _build_execution_context() -> Dict[str, Any]:
        resolved_content = RemoteExecutionTool._resolve_model_content_list()
        return {
            "model_list": RemoteExecutionTool.global_memory.memory_retrieve("model_list") or [],
            "model_id_list": RemoteExecutionTool.global_memory.memory_retrieve("model_id_list") or [],
            "model_url_list": RemoteExecutionTool.global_memory.memory_retrieve("model_url_list") or [],
            "github_url": RemoteExecutionTool.global_memory.memory_retrieve("github_url") or [],
            "family_md": RemoteExecutionTool.global_memory.memory_retrieve("family_md") or "",
            "family_index_js": RemoteExecutionTool.global_memory.memory_retrieve("family_index_js") or "",
            "family_content": RemoteExecutionTool.global_memory.memory_retrieve("family_content") or "",
            "model_content_list": resolved_content["model_content_list"],
            "content_resolution_warning": resolved_content["warning"],
            "execution_result": RemoteExecutionTool.global_memory.memory_retrieve("execution_result") or [],
            "executed_command": RemoteExecutionTool.global_memory.memory_retrieve("executed_command") or [],
            "fail_reason_list": RemoteExecutionTool.global_memory.memory_retrieve("fail_reason_list") or [],
            "ssh_config": RemoteExecutionTool.global_memory.memory_retrieve("ssh_config") or {},
        }

    @tool("memory_retrieve_execution_context")
    def memory_retrieve_execution_context():
        """Retrieve all information needed for remote execution from GLOBAL_MEMORY.
        Returns: dictionary containing model_id_list, per-model model_content_list and request config."""
        return RemoteExecutionTool._build_execution_context()

    @tool("memory_preview_remote_content")
    def memory_preview_remote_content(preview_chars: int = 1000) -> Dict[str, Any]:
        """Preview per-model content that will be sent to remote API before execution.
        Returns model-level summaries including content length and truncated preview."""
        context = RemoteExecutionTool._build_execution_context()
        model_ids = context.get("model_id_list") or []
        model_contents = context.get("model_content_list") or []
        preview_items: List[Dict[str, Any]] = []

        for idx, model_id in enumerate(model_ids):
            content = model_contents[idx] if idx < len(model_contents) else ""
            preview_items.append(
                {
                    "idx": idx,
                    "model_id": model_id,
                    "content_length": len(content or ""),
                    "content_preview": (content or "")[: max(0, int(preview_chars))],
                }
            )

        return {
            "count": len(preview_items),
            "content_resolution_warning": context.get("content_resolution_warning", ""),
            "items": preview_items,
        }

    @tool("execute_remote_readme_validation")
    def execute_remote_readme_validation(model_id: str, model_content: str) -> Dict[str, Any]:
        """Validate single-model content (md+js) on remote local-bkc agent by one HTTP request.
        Inputs: model_id, model_content
        Returns: remote validation result payload."""
        request_url = RemoteExecutionTool._resolve_request_url()
        ssh_config = RemoteExecutionTool.global_memory.memory_retrieve("ssh_config") or {}
        extra_payload = ssh_config.get("request_payload", {})
        request_stream = bool(ssh_config.get("request_stream", False))
        include_extracted_commands = bool(ssh_config.get("include_extracted_commands", False))
        return RemoteExecutionTool.client.validate_model_readme(
            request_url=request_url,
            model_id=model_id,
            content=model_content,
            extra_payload=extra_payload,
            stream=request_stream,
            include_extracted_commands=include_extracted_commands,
        )

    @tool("memory_store_execution_result")
    def memory_store_execution_result(
        idx: int,
        command_str: str,
        result: Any,
        fail_reason: Optional[str],
        updated_readme: Optional[str],
    ):
        """Store remote execution results into GLOBAL_MEMORY at specified index.
        Inputs: index, executed command, execution result, fail reason, updated readme
        Returns: success message."""

        def update_list(key, value):
            lst = RemoteExecutionTool.global_memory.memory_retrieve(key=key) or []
            while len(lst) <= idx:
                lst.append(None)
            lst[idx] = value
            RemoteExecutionTool.global_memory.memory_store(key=key, value=lst)

        update_list("executed_command", command_str)
        stored_result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        update_list("execution_result", stored_result)
        update_list("fail_reason_list", fail_reason)
        return True
