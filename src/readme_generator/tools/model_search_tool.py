import json
import os
import requests
from typing import List
from crewai import Agent,Task
from crewai.tools import tool
# from langchain.tools import tool
from .web_tool import backup_proxy_in_process,clear_proxy_in_process,restore_proxy_in_process

class ModelSearchTool:
    @tool("Search Model")
    def huggingface_model_search_url(model_list:List[str]):
        """Retrieve the relevant URL from HuggingFace based on the model_name."""
        proxy_backup={
            "http_proxy":"http://proxy.ims.intel.com:911",
            "https_proxy":"https://proxy.ims.intel.com:911"
        }
        proxy_backup=restore_proxy_in_process(proxy_backup=proxy_backup)
        API_URL="https://huggingface.co/api/models"
        model_url_list=[]
        model_id_list=[]
        for model_name in model_list:
            params={
                "search":model_name,
                "sort":"downloads",
                "direction":"-1",
                "limit":1
            }
            try:
                rsp=requests.get(API_URL,params=params,timeout=15)
                rsp.raise_for_status()
                models=rsp.json()
                if models:
                    model_id=models[0]["modelId"]
                    model_id_list.append(model_id)
                    model_url_list.append(f"https://huggingface.co/{model_id}")
                else:
                    model_id_list.append("")
                    model_url_list.append("")
            except Exception as e:
                print(f"查询失败:{e}")
                model_id_list.append("")
                model_url_list.append("")
        proxy_backup=clear_proxy_in_process(proxy_backup=proxy_backup)
        return model_url_list,model_id_list
    
    @tool("Search HF-mirror Model")
    def huggingface_mirror_model_search_url(model_list:List[str]):
        """Retrieve the relevant URL from HuggingFace mirror based on the model_name."""
        proxy_backup={
            "http_proxy":"http://proxy-dmz.intel.com:912",
            "https_proxy":"http://proxy-dmz.intel.com:912"
        }
        proxy_backup=restore_proxy_in_process(proxy_backup=proxy_backup)
        API_URL="https://hf-mirror.com/api/models"
        model_url_list=[]
        model_id_list=[]
        for model_name in model_list:
            params={
                "search":model_name,
                "sort":"downloads",
                "direction":"-1",
                "limit":1
            }
            try:
                rsp=requests.get(API_URL,params=params,timeout=15)
                rsp.raise_for_status()
                models=rsp.json()
                if models:
                    model_id=models[0]["modelId"]
                    model_id_list.append(model_id)
                    model_url_list.append(f"https://hf-mirror.com/{model_id}")
                else:
                    model_id_list.append("")
                    model_url_list.append("")
            except Exception as e:
                print(f"查询失败:{e}")
                model_id_list.append("")
                model_url_list.append("")
        proxy_backup=clear_proxy_in_process(proxy_backup=proxy_backup)
        return model_url_list,model_id_list
        