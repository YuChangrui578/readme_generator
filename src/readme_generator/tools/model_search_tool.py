import json
import os
import requests

from crewai import Agent,Task
from langchain.tool import tool

class ModelSearchTool():
    
    @tool("根据model_name从huggingface上获取相关的url")
    def huggingface_model_search_url(model_name:str):
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
                return f"https://hf-mirror.com/{model_id}"
            return None
        except Exception as e:
            print(f"查询失败:{e}")
            return None
        