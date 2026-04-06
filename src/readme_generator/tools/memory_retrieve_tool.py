import json
from typing import List,Dict
from crewai_tools import BaseTool
from pydantic import BaseModel,Field
from langchain.tools import tool
from MemoryStoreTool import GLOBAL_MEMORY

class RetrieveInput(BaseModel):
    query:str=Field(default="all",description="Description of what data to retrieve (e.g., 'all models', 'failed tests only')")

class MemoryRetrieveTool(BaseTool):
    @tool("Retrieves stored model data from shared memory.\
        Returns a raw list of all stored entries if query is 'all'.\
        Use this before merging.")
    def retrieve_memory(query:str="all")->str:
        if not GLOBAL_MEMORY:
            return "No data found in memory"
        return json.dumps(GLOBAL_MEMORY,ensure_ascii=False,indent=2)

