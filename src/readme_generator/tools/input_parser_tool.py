from crewai.llm import LLM
from crewai.tools import tool
import json
from typing import Dict,Any,List
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
    def parse(cls,input_text:str,key_list:List[str],value_type:List[str])->Dict[str,Any]:
        schema_str="key:"+",".join(key_list)+"\n"+"value_type:"+",".join(value_type)
        prompt=f"""
You are a structured input parser.
Extract values from input_text according to the GLOBAL MEMORY SCHEMA.
ONLY output a JSON object, NO extra words.
If a key's value is NOT found in input_text, DO NOT output it.

GLOBAL MEMORY SCHEMA:
{schema_str}

CRITICAL INSTRUCTIONS for 'github_url':
1. 'github_url' MUST be a LIST of strings, not a single string.
2. The length of the 'github_url' list MUST match the length of the 'model_list'.
3. The order of URLs in 'github_url' must correspond exactly to the order of models in 'model_list'.
4. If a model has no dedicated GitHub repository (e.g., uses official CPU env), use an empty string "" as its placeholder.

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

            # 安全过滤：只保留 schema 中存在的 key
            cleaned = {}
            for k, t in zip(key_list, value_type):
                if k in parsed:
                    cleaned[k] = parsed[k]

            return cleaned

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            return {}
        
class InputParseTool():

    @tool("get_input_text")
    def get_input_text():
        """Get input text from Global Memory"""
        memory=GlobalMemory()
        input_text=memory.memory_retrieve("input_text")
        return input_text

    @tool("parse_input_text")
    def parse_input_text(input_text:str,key_list:List[str],value_type:List[str])->Dict[str,Any]:
        """Parse input_text to extract structured model information.
        Extracts model_list ,github_url etc.
        Returns a dictionary containing parsed structured data."""
        return InternelParserLLM.parse(input_text=input_text,key_list=key_list,value_type=value_type)

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
        data=json.loads(data)
        memory=GlobalMemory()

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