from crewai.tools import tool
from typing import List,Optional
from crewai.llm import LLM
from .memory_tool import GlobalMemory
from .chatopenai import LLM_Callable

class ReadmeMergeClient:
    def __init__(self):
        self.llm=LLM_Callable(
            base_url="http://10.54.34.78:30000/v1",
            api_key="empty",
            model_name="local-model"
        )

    def merge_series_readme(self,model_list,readme_list,merge_template):
        prompt = f"""
Merge all individual model READMEs into one unified model series README.
Follow the structure strictly from the reference example.
Highlight precision differences (fp8/int4/awq etc.)

Model series: {model_list}

Individual READMEs:
{readme_list}

Reference template:
{merge_template}

Output only the final merged README.
        """
        result= self.llm.invoke(prompt).strip()
        import re
        result=re.sub(r"<think>.*?</think>","",result,flags=re.DOTALL)
        return result.strip()

class MergeReadmeTool:
    merger=ReadmeMergeClient()

    @tool("memory_retrieve_merge_context")
    def memory_retrieve_merge_context():
        """Retrieve all information needed for README merging from GLOBAL_MEMORY.
        Returns: dictionary containing model_list, model_readme, merged_reference_example.
        """
        memory=GlobalMemory()
        return {
            "model_list":memory.memory_retrieve("model_list"),
            "model_readme":memory.memory_retrieve("model_readme"),
            "merged_reference_example":memory.memory_retrieve("merged_reference_example")
        }
    
    @tool("merge_model_series_readme")
    def merge_model_series_readme(model_list:List[str],model_readme_list:List[str],merge_template:List[str])->str:
        """Merge multiple individual model READMEs into one unified model series README.
        Inputs: model_list, model_readme_list, merged_reference_example (template)
        Returns: merged README content as a single string."""
        return MergeReadmeTool.merger.merge_series_readme(
            model_list=model_list,
            model_readme_list=model_readme_list,
            merge_template=merge_template
        )
    
    @tool("memory_store_merged_readme")
    def memory_store_merged_readme(merged_readme:str):
        """Store merged README into GLOBAL_MEMORY with key "merged_readme".
        Input: merged README content string
        Returns: success message."""
        memory=GlobalMemory()
        memory.memory_store("merged_readme",merged_readme)
        return True
    
