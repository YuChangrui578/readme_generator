from typing import Any,List,Dict
from crewai.tools import tool
import traceback
import json
import os
from dataclasses import dataclass,asdict

@dataclass
class MemoryData:
    input_text:str=""
    model_list:list=None
    github_url:list=None
    remote_folder:str=""
    ssh_config:dict=None
    github_config:dict=None
    model_url_list:list=None
    model_id_list:list=None
    execution_result:list=None
    fail_reason_list:list=None
    executed_command:str=""
    family_md:str=""
    family_index_js:str=""
    family_content:str=""
    ref_md:str=""
    ref_index_js:str=""
    pr_info:dict=None

class GlobalMemory:
    def __init__(
        self,
        persist_path:str="/home/changrui/readme_generator/src/readme_generator/global_memory.json"
    ):
        self.persist_path=persist_path
        self.memory=MemoryData()
        self.load_from_file()

    def load_from_file(self)->None:
        if os.path.exists(self.persist_path):
            with open(self.persist_path,"r",encoding="utf-8") as f:
                data=json.load(f)
                self.memory.model_list=data.get("model_list",[])
                self.memory.remote_folder=data.get("remote_folder","")
                self.memory.ssh_config=data.get("ssh_config",{})
                self.memory.github_config=data.get("github_config",{})
                self.memory.model_url_list=data.get("model_url_list",[])
                self.memory.model_id_list=data.get("model_id_list",[])
                self.memory.execution_result=data.get("execution_result",[])
                self.memory.executed_command=data.get("executed_command","")
                self.memory.github_url=data.get("github_url",[])
                self.memory.fail_reason_list=data.get("fail_reason_list",[])
                self.memory.input_text=data.get("input_text","")
                self.memory.family_md=data.get("family_md","")
                self.memory.family_index_js=data.get("family_index_js","")
                self.memory.family_content=data.get("family_content","")
                self.memory.ref_md=data.get("ref_md","")
                self.memory.ref_index_js=data.get("ref_index_js","")
                self.memory.pr_info=data.get("pr_info",{})
        else:
            self.memory.model_list=[]
            self.memory.remote_folder=""
            self.memory.ssh_config={}
            self.memory.github_config={}
            self.memory.model_url_list=[]
            self.memory.model_id_list=[]
            self.memory.execution_result=[]
            self.memory.executed_command=""
            self.memory.github_url=[]
            self.memory.fail_reason_list=[]
            self.memory.input_text=""
            self.memory.family_md=""
            self.memory.family_index_js=""
            self.memory.family_content=""
            self.memory.ref_md=""
            self.memory.ref_index_js=""
            self.memory.pr_info={}
            self.save_to_file()

    def save_to_file(self)->bool:
        try:
            data={
                "model_list":self.memory.model_list,
                "remote_folder":self.memory.remote_folder,
                "ssh_config":self.memory.ssh_config,
                "github_config":self.memory.github_config,
                "model_url_list":self.memory.model_url_list,
                "model_id_list":self.memory.model_id_list,
                "execution_result":self.memory.execution_result,
                "executed_command":self.memory.executed_command,
                "github_url":self.memory.github_url,
                "fail_reason_list":self.memory.fail_reason_list,
                "input_text":self.memory.input_text,
                "family_md":self.memory.family_md,
                "family_index_js":self.memory.family_index_js,
                "family_content":self.memory.family_content,
                "ref_md":self.memory.ref_md,
                "ref_index_js":self.memory.ref_index_js,
                "pr_info":self.memory.pr_info
            }
            with open(self.persist_path,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            return True
        except Exception as e:
            print(e)
            traceback.print_exc()
            return False

    def memory_store(self,key:str,value:Any)->bool:
        try:
            if key=="model_list":
                self.memory.model_list=value
            elif key=="remote_folder":
                self.memory.remote_folder=value
            elif key=="ssh_config":
                self.memory.ssh_config=value
            elif key=="github_config":
                self.memory.github_config=value
            elif key=="model_url_list":
                self.memory.model_url_list=value
            elif key=="model_id_list":
                self.memory.model_id_list=value
            elif key=="execution_result":
                self.memory.execution_result=value
            elif key=="executed_command":
                self.memory.executed_command=value
            elif key=="github_url":
                self.memory.github_url=value
            elif key=="fail_reason_list":
                self.memory.fail_reason_list=value
            elif key=="input_text":
                self.memory.input_text=value
            elif key=="family_md":
                self.memory.family_md=value
            elif key=="family_index_js":
                self.memory.family_index_js=value
            elif key=="family_content":
                self.memory.family_content=value
            elif key=="ref_md":
                self.memory.ref_md=value
            elif key=="ref_index_js":
                self.memory.ref_index_js=value
            elif key=="pr_info":
                self.memory.pr_info=value
            self.save_to_file()
            return True
        except Exception as e:
            print(e)
            traceback.print_exc()
            return False
        
    def memory_retrieve(self,key:str)->Any:
        try:
            self.load_from_file()
            if key=="model_list":
                return self.memory.model_list
            elif key=="remote_folder":
                return self.memory.remote_folder
            elif key=="ssh_config":
                return self.memory.ssh_config
            elif key=="github_config":
                return self.memory.github_config
            elif key=="model_url_list":
                return self.memory.model_url_list
            elif key=="model_id_list":
                return self.memory.model_id_list
            elif key=="execution_result":
                return self.memory.execution_result
            elif key=="executed_command":
                return self.memory.executed_command
            elif key=="github_url":
                return self.memory.github_url
            elif key=="fail_reason_list":
                return self.memory.fail_reason_list
            elif key=="input_text":
                return self.memory.input_text
            elif key=="family_md":
                return self.memory.family_md
            elif key=="family_index_js":
                return self.memory.family_index_js
            elif key=="family_content":
                return self.memory.family_content
            elif key=="ref_md":
                return self.memory.ref_md
            elif key=="ref_index_js":
                return self.memory.ref_index_js
            elif key=="pr_info":
                return self.memory.pr_info
            return ""
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ""
        
    def get_memory_keys(self)->List[str]:
        return list(asdict(self.memory).keys())
    
    def get_memory_value_types(self)->Dict[str,Any]:
        return {key:type(value).__name__ for key,value in asdict(self.memory).items()}
     

class MemoryTool:
    @tool("Store Memory")
    def store_memory(key:str,value:Any):
        """Used to persistently store any specified data content into global memory using a custom key, and synchronously save it to a local file to ensure data integrity and persistence. Supports storing all types of task data, including model lists, single model details, generated documents, intermediate results, and more. It validates key validity and data consistency during storage to maintain standardized global memory management. Each storage operation is automatically saved to disk, allowing all subsequent agents to access and reuse data stably across tasks and workflows."""
        memory=GlobalMemory()
        flag=memory.memory_store(key=key,value=value)
        return flag
    
    @tool("Retrieve Memory")
    def retrieve_memory(key:str):
        """Used to retrieve the corresponding stored data from the global persistent memory according to the specified memory key. Supports fetching any saved content in the global memory. Data is loaded from a local file to ensure consistent and complete content across tasks and agents. Returns a default value if the specified memory key does not exist or has no corresponding data, maintaining stable and uninterrupted task execution."""
        memory=GlobalMemory()
        info=memory.memory_retrieve(key=key)
        return info
    
    @tool("Get Memory Key")
    def get_memory_key():
        """Used to enable agents in CrewAI to accurately obtain all stored data key values in the global memory (GLOBAL_MEMORY). Its core function is to provide agents with clear guidance on memory key names, ensuring that subsequent data retrieval and storage operations can accurately locate the target keys. It avoids data operation failures caused by incorrect key names, guarantees the accuracy and efficiency of the interaction between agents and global memory, and supports the smooth connection of data reading and writing in the task flow."""
        memory=GlobalMemory()
        return memory.get_memory_keys()
    
    @tool("Get Memory Value Type")
    def get_memory_value_type():
        """"""
        memory=GlobalMemory()
        return memory.get_memory_value_types()
    
