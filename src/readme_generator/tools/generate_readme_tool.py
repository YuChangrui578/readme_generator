from crewai.tools import tool
from typing import List,Optional
from crewai.llm import LLM
from .memory_tool import GlobalMemory
from .chatopenai import LLM_Callable

class ReadmeGenerationClient:
    def __init__(self):
        # self.llm=LLM(
        #     model="local-model",
        #     base_url="http://10.54.34.78:3000/v1",
        #     api_key="empty"
        # )
        self.llm=LLM_Callable(
            base_url="http://10.54.34.78:30000/v1",
            api_key="empty",
            model_name="local-model"
        )

    def fetch_template(self,url):
        pass

    def render_readme(self,model_name,model_id,model_url,template):
        prompt=f"""
        You are a professional README writer.
        Use the template to generate standardized README.
        Replace:
        - model_name: {model_name}
        - model_id: {model_id}
        - model_url: {model_url}

        Template:
        {template}
        """
        result= self.llm.invoke(prompt).strip()
        import re
        result=re.sub(r"<think>.*?</think>","",result,flags=re.DOTALL)
        return result.strip()
    


class GenerateReadmeTool:
    generator=ReadmeGenerationClient()

    @tool("get_refernce_example_list")
    def get_reference_example_list(github_url:List[str],origin_reference_example_list:List[str])->List[str]:
        """Get reference README examples from github_url list.
        Input: list of github URLs
        Returns: list of reference README contents, one-to-one index correspondence with model_list."""
        reference_example_list=[]
        memory=GlobalMemory()
        for u in github_url:
            if len(u)>0:
                reference_example_list.append(origin_reference_example_list[1])
            else:
                reference_example_list.append(origin_reference_example_list[0])
        memory.memory_store("reference_example_list",reference_example_list)
        return reference_example_list
        
    @tool("memory_retrieve_model_all_info")
    def memory_retrieve_model_all_info():
        """Retrieve all model-related information from GLOBAL_MEMORY.
        Returns: dictionary containing model_list, model_id_list, model_url_list."""
        memory=GlobalMemory()
        return {
            "model_list":memory.memory_retrieve("model_list"),
            "model_id_list":memory.memory_retrieve("model_id_list"),
            "model_url_list":memory.memory_retrieve("model_url_list"),
            "github_url":memory.memory_retrieve("github_url"),
            "origin_reference_example_list":memory.memory_retrieve("origin_reference_example_list")
        }
    
    @tool("batch_generate_model_readme")
    def batch_generate_model_readme(model_list:List[str],model_id_list:List[str],model_url_list:List[str],reference_example_list:List[str])->List[str]:
        """ Batch generate model README documents based on reference templates.
        Inputs: model_list, model_id_list, model_url_list, reference_example_list
        Returns: list of generated README contents, one-to-one index correspondence."""
        readme_list=[]
        count=min(len(model_list),len(reference_example_list))
        for i in range(count):
            print(i)
            name=model_list[i]
            mid=model_id_list[i] if i<len(model_id_list) else ""
            url=model_url_list[i] if i<len(model_url_list) else ""
            tpl=reference_example_list[i] if i<len(reference_example_list) else ""
            readme=GenerateReadmeTool.generator.render_readme(name,mid,url,tpl)
            readme_list.append(readme)
        return readme_list
    
    @tool("memory_store_model_readme")
    def memory_store_model_readme(model_readme_list:List[str]):
        """Store generated model README list into GLOBAL_MEMORY with key "model_readme".
        Input: list of generated README contents
        Returns: success message."""
        memory=GlobalMemory()
        memory.memory_store("model_readme",model_readme_list)
        return True
    

