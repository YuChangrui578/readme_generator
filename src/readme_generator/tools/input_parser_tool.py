from crewai.llm import LLM
from crewai.tools import tool
import json
from typing import Dict,Any,List
import re
import ast
from .memory_tool import GlobalMemory
import traceback
from .chatopenai import LLM_Callable

class InternelParserLLM:
    llm=LLM_Callable(
            base_url="http://10.54.34.78:30000/v1",
            api_key="empty",
            model_name="local-model"
        )

    @classmethod
    def _fallback_parse(cls, input_text: str) -> Dict[str, Any]:
        text = input_text or ""
        github_url = []
        for url in re.findall(r"https?://github\.com/[^\s,\]\)\"']+", text, flags=re.IGNORECASE):
            cleaned = url.rstrip(".,;:!?)")
            if cleaned not in github_url:
                github_url.append(cleaned)

        model_list = []
        for m in re.findall(r"\b[A-Za-z][A-Za-z0-9_.]*(?:-[A-Za-z0-9_.]+)+\b", text):
            ml = m.lower()
            if any(k in ml for k in ("llama", "mistral", "gemma", "qwen", "deepseek", "phi")):
                if m not in model_list:
                    model_list.append(m)

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
Extract values from input_text.
ONLY output a JSON object, NO extra words.
Output schema:
{{
  "model_list": ["..."],
  "github_url": ["..."]
}}

CRITICAL INSTRUCTIONS for 'github_url':
1. 'github_url' MUST be a LIST of strings, not a single string.
2. Keep only GitHub repository URLs.
3. The length of github_url MUST match model_list.
4. Use empty string "" placeholders for official-sglang models.
5. If only one dev-branch URL exists for a model family, keep it at the last model and fill previous entries with "".

Input text:
{input_text}

Output ONLY valid JSON:
"""
        try:
            response = cls.llm.invoke(prompt)
            import re
            response=re.sub(r"<think>.*?</think>","",response,flags=re.DOTALL)
            response = response.strip()
            parsed = json.loads(response.strip())

            cleaned = {
                "model_list": parsed.get("model_list", []),
                "github_url": parsed.get("github_url", []),
            }
            if not isinstance(cleaned["model_list"], list):
                cleaned["model_list"] = []
            if not isinstance(cleaned["github_url"], list):
                cleaned["github_url"] = []

            return cleaned

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return cls._fallback_parse(input_text)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            return cls._fallback_parse(input_text)
        
class InputParseTool():
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
        memory=GlobalMemory()
        input_text=memory.memory_retrieve("input_text")
        return input_text

    @tool("parse_input_text")
    def parse_input_text(input_text:str)->Dict[str,Any]:
        """Parse input_text to extract structured model information.
        Extracts model_list ,github_url etc.
        Returns a dictionary containing parsed structured data."""
        parsed = InternelParserLLM.parse(input_text=input_text)
        model_list = parsed.get("model_list", [])
        github_url = parsed.get("github_url", [])
        parsed["github_url"] = InputParseTool._align_github_url(model_list, github_url)
        return parsed

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
        memory=GlobalMemory()

        model_list = data.get("model_list", memory.memory_retrieve("model_list") or [])
        github_url = data.get("github_url", memory.memory_retrieve("github_url") or [])
        if isinstance(model_list, list) and isinstance(github_url, list):
            data["github_url"] = InputParseTool._align_github_url(model_list, github_url)

        for k,v in data.items():
            memory.memory_store(k,v)
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
