import json
import os
import requests

from crewai import Agent,Task
from crewai.tools import tool
# from langchain.tools import tool

class ModelSearchTool:
    @tool("Search Model")
    def huggingface_model_search_url(model_name:str):
        """Retrieve the relevant URL from Hugging Face based on the model_name."""
        API_URL="https://hf-mirror.com/api/models"
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
                return f"https://hf-mirror.com/{model_id}",model_id
            return None,None
        except Exception as e:
            print(f"查询失败:{e}")
            return None,None
        