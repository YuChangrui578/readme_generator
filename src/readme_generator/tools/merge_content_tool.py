from crewai_tools import BaseTool
from pydantic import BaseModel,Field
from langchain.tools import tool
from MemoryStoreTool import GLOBAL_MEMORY
import json

class MergeContentInput(BaseModel):
    series_name: str = Field(..., description="Name of the model series (e.g., 'Llama-3-Series')")
    introduction: str = Field(..., description="General introduction for the whole series")

class MergeContentTool():

    @tool("Retrieves all processed model segments from memory and merges them into a single \
        comprehensive README and a unified Test Report.\
        Input: series_name and a general introduction.\
        Output: A dictionary with keys 'final_readme' and 'final_test_report'.")
    def merge_readme(series_name:str,introduction:str)->dict:
        if not GLOBAL_MEMORY:
            return {"error":"No data to merge."}

        raw_data=json.dumps(GLOBAL_MEMORY,ensure_ascii=False,ensure_ascii=False,indent=2)
        return {
            "raw_data":raw_data,
            "series_name":series_name,
            "instructions":introduction,
            "count":len(GLOBAL_MEMORY)
        }

    