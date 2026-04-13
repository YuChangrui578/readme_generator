from pydantic import BaseModel,Field
from crewai.tools import tool
from .memory_tool import GlobalMemory
import json

class MergeReadmeTool():

    @tool("Merge README")
    def merge_readme(series_name:str,introduction:str)->dict:
        """Retrieves all processed model segments from memory and merges them into a single \
        comprehensive README and a unified Test Report.\
        Input: series_name and a general introduction.\
        Output: A dictionary with keys 'final_readme' and 'final_test_report'."""
        GLOBAL_MEMORY=GlobalMemory()
        if not GLOBAL_MEMORY:
            return {"error":"No data to merge."}

        raw_data=json.dumps(GLOBAL_MEMORY,ensure_ascii=False,indent=2)
        return {
            "raw_data":raw_data,
            "series_name":series_name,
            "instructions":introduction,
            "count":len(GLOBAL_MEMORY)
        }

    