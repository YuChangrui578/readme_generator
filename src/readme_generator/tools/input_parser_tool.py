from crewai.tools import tool
import json
from typing import Dict,Any,List
import ast
from .memory_tool import GlobalMemory
import traceback
from .chatopenai import LLM_Callable
from .common_utils import (
    filter_model_candidates,
    infer_family_hint_from_corpus,
    infer_models_from_corpus,
    is_url_source_mode,
    normalize_list,
)

class InternelParserLLM:
    llm=LLM_Callable(
            base_url="http://10.54.34.78:30000/v1",
            api_key="empty",
            model_name="local-model"
        )

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        s = str(text or "")
        start_tag = "<think>"
        end_tag = "</think>"
        while True:
            start = s.find(start_tag)
            if start < 0:
                break
            end = s.find(end_tag, start + len(start_tag))
            if end < 0:
                s = s[:start]
                break
            s = s[:start] + s[end + len(end_tag):]
        return s.strip()

    @classmethod
    def _fallback_parse(cls, input_text: str) -> Dict[str, Any]:
        text = input_text or ""
        github_url: List[str] = []
        for token in str(text).replace("\n", " ").split():
            candidate = token.strip(".,;:!?)，。；：！？）】》」』、\"'")
            if "github.com/" not in candidate.lower():
                continue
            if not candidate.startswith(("http://", "https://")):
                continue
            if candidate not in github_url:
                github_url.append(candidate)

        source_md_url = ""
        source_js_url = ""
        for url in github_url:
            lower = url.lower()
            if (lower.endswith(".md") or "/docs/" in lower or "/readme" in lower) and not source_md_url:
                source_md_url = url
            if ("/src/" in lower or "configgenerator" in lower or lower.endswith(".js") or lower.endswith("/")) and not source_js_url:
                source_js_url = url

        if source_md_url and source_js_url:
            return {
                "generation_mode": "web_sources",
                "source_md_url": source_md_url,
                "source_js_url": source_js_url,
                "model_list": [],
                "github_url": [],
            }

        family_hint = infer_family_hint_from_corpus([text])
        model_list = infer_models_from_corpus([text], family_hint=family_hint)
        model_list = filter_model_candidates(model_list, family_hint=family_hint)

        return {
            "model_list": model_list,
            "github_url": github_url,
        }

    @classmethod
    def _extract_from_workflow_payload(cls, input_text: str) -> Dict[str, Any]:
        text = (input_text or "").strip()
        if not text:
            return {}

        parsed_obj = None
        for parser in (
            lambda s: json.loads(s),
            lambda s: ast.literal_eval(s),
        ):
            try:
                parsed_obj = parser(text)
                break
            except Exception:
                continue

        if not isinstance(parsed_obj, dict):
            return {}

        generation_mode = str(parsed_obj.get("generation_mode") or "").strip().lower()
        source_md_url = str(
            parsed_obj.get("source_md_url")
            or parsed_obj.get("github_md_folder_url")
            or ""
        ).strip()
        source_js_url = str(
            parsed_obj.get("source_js_url")
            or parsed_obj.get("github_js_folder_url")
            or ""
        ).strip()
        if source_md_url or source_js_url:
            # URL-source mode: only official sglang path for now, no dev branch url list.
            return {
                "generation_mode": generation_mode or "web_sources",
                "source_md_url": source_md_url,
                "source_js_url": source_js_url,
                "model_list": parsed_obj.get("model_list", []) if isinstance(parsed_obj.get("model_list"), list) else [],
                "github_url": [],
            }

        model_list = parsed_obj.get("model_list")
        github_url = parsed_obj.get("github_url")
        if isinstance(model_list, list) and isinstance(github_url, list):
            return {"model_list": model_list, "github_url": github_url}

        # Old/incorrect payload shape fallback: {input_text, key_list, value_type}
        embedded = parsed_obj.get("input_text")
        if isinstance(embedded, str):
            return cls._fallback_parse(embedded)
        return {}

    @classmethod
    def parse(cls,input_text:str)->Dict[str,Any]:
        extracted = cls._extract_from_workflow_payload(input_text)
        if extracted:
            return extracted

        prompt=f"""
You are a structured input parser.
Extract values from arbitrary human text (not necessarily JSON).
ONLY output a JSON object, NO extra words.
Output schema:
{{
  "generation_mode": "reference|web_sources",
  "source_md_url": "",
  "source_js_url": "",
  "model_list": ["..."],
  "github_url": ["..."]
}}

Rules:
1. If both source_md_url and source_js_url exist (github/web), set generation_mode="web_sources".
2. In web_sources mode, github_url MUST be [] (official sglang only).
3. If no source urls are found, set generation_mode="reference".
4. Extract model names from free text (examples: "qwen3 7b", "Llama 3.1 8B Instruct", "Qwen3-8B").
5. Canonicalize model names with '-' when possible (e.g., Qwen3-7B, Llama-3.1-8B-Instruct).
6. Keep model_list de-duplicated.
7. Keep only GitHub repository URLs in github_url for legacy mode.

CRITICAL INSTRUCTIONS for legacy 'github_url':
1. 'github_url' MUST be a LIST of strings, not a single string.
2. The length of github_url SHOULD match model_list.
3. Use empty string "" placeholders for official-sglang models when uncertain.
4. If only one dev-branch URL exists for a model family, keep it at the last model and fill previous entries with "".

Input text:
{input_text}

Output ONLY valid JSON:
"""
        try:
            response = cls.llm.invoke(prompt, timeout=120)
            response = cls._strip_think_blocks(response)
            parsed = json.loads(response.strip())

            cleaned = {
                "generation_mode": str(parsed.get("generation_mode") or "").strip().lower(),
                "source_md_url": str(parsed.get("source_md_url") or "").strip(),
                "source_js_url": str(parsed.get("source_js_url") or "").strip(),
                "model_list": parsed.get("model_list", []),
                "github_url": parsed.get("github_url", []),
            }
            if cleaned["generation_mode"] not in {"reference", "web_sources", "github_folders", "url_source"}:
                cleaned["generation_mode"] = ""
            if not isinstance(cleaned["model_list"], list):
                cleaned["model_list"] = []
            if not isinstance(cleaned["github_url"], list):
                cleaned["github_url"] = []
            if cleaned["source_md_url"] and cleaned["source_js_url"] and not cleaned["generation_mode"]:
                cleaned["generation_mode"] = "web_sources"

            return cleaned

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return cls._fallback_parse(input_text)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            return cls._fallback_parse(input_text)
        
class InputParseTool():
    global_memory = None

    @staticmethod
    def _memory() -> GlobalMemory:
        return InputParseTool.global_memory or GlobalMemory()

    @staticmethod
    def _build_remote_payload(memory: GlobalMemory, mode: str, model_list: List[str], github_url: List[str]) -> Dict[str, Any]:
        if is_url_source_mode(mode):
            source_md = str(memory.memory_retrieve("source_md_url") or "").strip()
            source_js = str(memory.memory_retrieve("source_js_url") or "").strip()
            return {
                "generation_mode": "url_source",
                "model_list": model_list or [],
                "source_urls": {"md": source_md, "js": source_js},
                "metadata": {"official_only": True},
            }

        official_models: List[str] = []
        dev_models: List[str] = []
        model_to_url: Dict[str, str] = {}
        for idx, m in enumerate(model_list or []):
            url = github_url[idx] if idx < len(github_url) else ""
            model_to_url[m] = url
            if str(url).strip():
                dev_models.append(m)
            else:
                official_models.append(m)
        return {
            "generation_mode": "legacy",
            "content": {
                "family_md": str(memory.memory_retrieve("family_md") or "") or str(memory.memory_retrieve("ref_md") or ""),
                "family_index_js": str(memory.memory_retrieve("family_index_js") or "") or str(memory.memory_retrieve("ref_index_js") or ""),
                "input_text": str(memory.memory_retrieve("input_text") or ""),
            },
            "model_list": model_list or [],
            "github_url": github_url or [],
            "metadata": {
                "official": official_models,
                "dev": dev_models,
                "model_to_github_url": model_to_url,
            },
        }

    @staticmethod
    def _align_github_url(model_list: List[Any], github_url: List[Any]) -> List[str]:
        models = [str(x) for x in (model_list or [])]
        urls = [str(x) for x in (github_url or [])]
        n = len(models)
        if n == 0:
            return urls
        if len(urls) == n:
            return urls
        if len(urls) == 0:
            return [""] * n
        if len(urls) == 1 and n > 1:
            # Common case: only one dev branch URL for the last (special) variant.
            return [""] * (n - 1) + urls
        if len(urls) < n:
            return urls + [""] * (n - len(urls))
        return urls[:n]

    @tool("get_input_text")
    def get_input_text():
        """Get input text from Global Memory"""
        memory=InputParseTool._memory()
        input_text=memory.memory_retrieve("input_text")
        return input_text

    @tool("parse_input_text")
    def parse_input_text(input_text:str)->Dict[str,Any]:
        """Parse input_text to extract structured model information.
        Extracts model_list ,github_url etc.
        Returns a dictionary containing parsed structured data."""
        parsed = InternelParserLLM.parse(input_text=input_text)
        mode = str(parsed.get("generation_mode") or "").strip().lower()
        if is_url_source_mode(mode):
            parsed["github_url"] = []
            parsed["generation_mode"] = "web_sources"
        else:
            model_list = normalize_list(parsed.get("model_list"), fallback_single_str=True, stringify_items=True)
            github_url = normalize_list(parsed.get("github_url"), fallback_single_str=True, stringify_items=True)
            parsed["github_url"] = InputParseTool._align_github_url(model_list, github_url)
            parsed["generation_mode"] = "reference"
        return parsed

    @staticmethod
    def _infer_model_list_with_llm(corpus: List[str], family_hint: str) -> List[str]:
        preview = "\n\n".join(str(x or "")[:4000] for x in corpus if str(x or "").strip())[:24000]
        if not preview:
            return []
        prompt = f"""
You are extracting concrete model variants for deployment docs.
Given mixed markdown and index.js content, infer model_list.

Rules:
1. Focus on one family when hint exists: "{family_hint}".
2. Use index.js labels/options together with md family context (e.g. label "7B" + family "Qwen3" => "Qwen3-7B").
3. Keep concrete variants only (size/variant specific), not URLs/paths/docs labels.
4. Output ONLY JSON: {{"model_list": ["..."]}}.

Content:
{preview}
"""
        try:
            response = InternelParserLLM.llm.invoke(prompt)
            response = InternelParserLLM._strip_think_blocks(response)
            parsed = json.loads(response)
            out = normalize_list(parsed.get("model_list"), fallback_single_str=False, stringify_items=True)
            return [x for x in out if x.strip()]
        except Exception:
            return []

    @staticmethod
    def _infer_model_list_from_source_context_memory(memory: GlobalMemory) -> List[str]:
        source_md_files = normalize_list(memory.memory_retrieve("source_md_files") or [])
        source_js_files = normalize_list(memory.memory_retrieve("source_js_files") or [])
        ref_md = str(memory.memory_retrieve("ref_md") or "")
        ref_index_js = str(memory.memory_retrieve("ref_index_js") or "")
        input_text = str(memory.memory_retrieve("input_text") or "")
        source_md_url = str(memory.memory_retrieve("source_md_url") or "")
        source_js_url = str(memory.memory_retrieve("source_js_url") or "")

        corpus: List[str] = [input_text, ref_md, ref_index_js, source_md_url, source_js_url]
        for item in source_md_files + source_js_files:
            if isinstance(item, dict):
                corpus.append(str(item.get("content") or ""))
            else:
                corpus.append(str(item))
        family_hint = infer_family_hint_from_corpus([input_text, source_md_url, source_js_url]) or infer_family_hint_from_corpus(corpus)
        llm_models = InputParseTool._infer_model_list_with_llm(corpus, family_hint)
        regex_models = infer_models_from_corpus(corpus, family_hint=family_hint)
        return filter_model_candidates(llm_models + regex_models, family_hint=family_hint)

    @tool("infer_model_list_from_source_context")
    def infer_model_list_from_source_context() -> List[str]:
        """Infer model_list from URL-source markdown/js content plus input text context."""
        memory = InputParseTool._memory()
        inferred = InputParseTool._infer_model_list_from_source_context_memory(memory)
        if inferred:
            memory.memory_store("model_list", inferred)
        return inferred

    # @tool("memory_get_key_tool")
    # def memory_get_keys():
    #     """Get all key names from GLOBAL_MEMORY schema.
    #     Returns a list of all available keys in GLOBAL_MEMORY."""
    #     memory=GlobalMemory()
    #     return memory.get_memory_keys()
    
    # @tool("memory_get_value_type_tool")
    # def memory_get_type():
    #     """Get the data type of a specific key from GLOBAL_MEMORY schema.
    #     Input: key name (string)
    #     Returns: data type of the key (e.g., list, str, dict)"""
    #     memory=GlobalMemory()
    #     return memory.get_memory_value_types()
    
    @tool("write_structured_data_to_global_memory")
    def store_memory(data:str):
        """Store data to Global Memory"""
        if isinstance(data, str):
            data=json.loads(data)
        if not isinstance(data, dict):
            raise ValueError("data must be JSON string or dict")
        memory=InputParseTool._memory()

        model_list = normalize_list(data.get("model_list", memory.memory_retrieve("model_list") or []), fallback_single_str=True, stringify_items=True)
        github_url = normalize_list(data.get("github_url", memory.memory_retrieve("github_url") or []), fallback_single_str=True, stringify_items=True)
        mode = str(data.get("generation_mode", memory.memory_retrieve("generation_mode") or "")).strip().lower()
        if is_url_source_mode(mode):
            data["generation_mode"] = "web_sources"
            data["github_url"] = []
            current_models = normalize_list(data.get("model_list"), fallback_single_str=True, stringify_items=True)
            current_models = [m for m in current_models if m.strip()]
            if not current_models:
                inferred = InputParseTool._infer_model_list_from_source_context_memory(memory)
                if inferred:
                    data["model_list"] = inferred
            model_list = normalize_list(data.get("model_list", []), fallback_single_str=True, stringify_items=True)
            github_url = []
        else:
            data["generation_mode"] = "reference"
            github_url = InputParseTool._align_github_url(model_list, github_url)
            data["github_url"] = github_url

        for k,v in data.items():
            memory.memory_store(k,v)
        remote_payload = InputParseTool._build_remote_payload(
            memory=memory,
            mode=data.get("generation_mode", ""),
            model_list=model_list,
            github_url=github_url,
        )
        memory.memory_store("remote_payload", remote_payload)
        return True
    
    # @tool("set_github_config_to_memory")
    # def set_github_config_to_memory(github_config:dict):
    #     """Store the independently passed GITHUB_CONFIG parameter into GLOBAL_MEMORY.
    #     GITHUB_CONFIG should contain: github_token, repo_owner, repo_name, base_branch, head_branch, pr_title, pr_description, commit_message, path.
    #     Returns a success message."""
    #     memory=GlobalMemory()
    #     memory.memory_store("github_config",github_config)
    #     return "Stored github_config successfully"
    
    # @tool("set_github_config_to_memory")
    # def set_ssh_config_to_memory(ssh_config:dict):
    #     """Store the independently passed SSH_CONFIG parameter into GLOBAL_MEMORY.
    #     SSH_CONFIG should contain: hostname, port, user_name, password.
    #     Returns a success message."""
    #     memory=GlobalMemory()
    #     memory.memory_store("ssh_config",ssh_config)
    #     return "Stored ssh_config successfully"

    # @tool("set_remote_folder_to_memory")
    # def set_remote_folder_to_memory(remote_folder:str):
    #     """Store the independently passed REMOTE_FOLDER parameter into GLOBAL_MEMORY.
    # REMOTE_FOLDER should contain the path of the remote folder.
    # Returns a success message."""
    #     memory=GlobalMemory()
    #     memory.memory_store("remote_folder",remote_folder)
    #     return "Stored remote_folder successfully"
    
    # @tool("set_origin_reference_example_list_to_memory")
    # def set_origin_reference_example_list_to_memory(origin_reference_example_list:List[str]):
    #     """Store the independently passed ORIGIN_REFERENCE_EXAMPLE_LIST parameter into GLOBAL_MEMORY.
    # ORIGIN_REFERENCE_EXAMPLE_LIST should contain a list of original reference examples.
    # Returns a success message."""
    #     memory=GlobalMemory()
    #     memory.memory_store("origin_reference_example_list_to_memory",origin_reference_example_list)
    #     return "Stored origin_reference_example_list successfully"
    
    # @tool("set_merged_reference_example")
    # def set_merged_reference_example_to_memory(merged_reference_example:str):
    #     """Store the independently passed MERGED_REFERENCE_EXAMPLE parameter into GLOBAL_MEMORY.
    # MERGED_REFERENCE_EXAMPLE should contain the merged reference example content.
    # Returns a success message."""
    #     memory=GlobalMemory()
    #     memory.memory_store("merged_reference_example",merged_reference_example)
    #     return "Store merged_reference_example successfully"
