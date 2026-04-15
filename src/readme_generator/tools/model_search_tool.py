import json
import os
import requests
from typing import List
from crewai import Agent,Task
from crewai.tools import tool
import traceback
# from langchain.tools import tool
from .web_tool import backup_proxy_in_process,clear_proxy_in_process,restore_proxy_in_process
from .memory_tool import GlobalMemory

class HuggingFaceModelClient:
    def search_model(sefl,name):
        try:
            params={"search":name,"sort":"downloads","direction":"-1","limit":1,}
            API_URL="https://huggingface.co/api/models"
            resp=requests.get(API_URL,params=params,timeout=10)
            resp.raise_for_status()
            models=resp.json()
            if not models:
                return None
            model_id=models[0]["modelId"]
            model_url=f"https://huggingface.co/{model_id}"
            return model_id,model_url
        except Exception as e:
            print(e)
            traceback.print_exc()
        
    def batch_search(self,names):
        model_ids=[]
        model_urls=[]
        for name in names:
            res=self.search_model(name=name)
            if res:
                mid,url=res
                model_ids.append(mid)
                model_urls.append(url)
            else:
                model_ids.append(None)
                model_urls.append(None)
        return {
            "model_id_list":model_ids,
            "model_url_list":model_urls
        }

class ModelSearchTool:
    hf_client=HuggingFaceModelClient()

    @tool("memory_retrieve_model_list")
    def memory_retrieve_model_list():
        """Retrieve model_list from GLOBAL_MEMORY.
        Returns: list of model names to search."""
        memory=GlobalMemory()
        return memory.memory_retrieve("model_list") or []
    
    @tool("huggingface_model_batch_search")
    def huggingface_model_batch_search(model_name_list:List[str])->List[str]:
        """Perform batch search for models on Hugging Face.
        Input: list of model names
        Returns: dictionary containing model_id_list and model_url_list, with one-to-one index correspondence."""
        return ModelSearchTool.hf_client.batch_search(model_name_list)
    
    @tool("memory_store_model_search_results")
    def memory_store_model_search_results(model_id_list:List[str],model_url_list:List[str]):
        """Store model search results (model_id_list and model_url_list) into GLOBAL_MEMORY.
        Inputs: model_id_list, model_url_list
        Returns: success message."""
        memory=GlobalMemory()
        memory.memory_store("model_id_list",model_id_list)
        memory.memory_store("model_url_list",model_url_list)
        return "Stored model search results successfully"