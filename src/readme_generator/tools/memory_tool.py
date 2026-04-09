from typing import Any,Optional,List,Dict
from crewai.tools import tool
import traceback
import json
import os
from dataclasses import dataclass,asdict

@dataclass
class MemoryData:
    model_list:list=None
    model_readme:list=None
    merged_readme:str=""
    remote_folder:str=""
    ssh_config:dict=None
    github_config:dict=None
    model_url_list:list=None
    model_id_list:list=None
    execution_result:list=None
    reference_example:str=""
    merged_readme_example:str=""
    executed_command:str=""

class GlobalMemory:
    def __init__(
        self,
        persist_path:str="global_memory.json"
    ):
        self.persist_path=persist_path
        self.memory=MemoryData()
        self.load_from_file()

    def load_from_file(self)->None:
        if os.path.exists(self.persist_path):
            with open(self.persist_path,"r",encoding="utf-8") as f:
                data=json.load(f)
                self.memory.model_list=data.get("model_list",[])
                self.memory.model_readme=data.get("model_readme",[])
                self.memory.merged_readme=data.get("merged_readme","")
                self.memory.remote_folder=data.get("remote_folder","")
                self.memory.ssh_config=data.get("ssh_config",{})
                self.memory.github_config=data.get("github_config",{})
                self.memory.model_url_list=data.get("model_url_list",[])
                self.memory.model_id_list=data.get("model_id_list",[])
                self.memory.execution_result=data.get("execution_result",[])
                self.memory.reference_example=data.get("reference_example","")
                self.memory.merged_reference_example=data.get("merged_reference_example","")
                self.memory.executed_command=data.get("executed_command","")
        else:
            self.memory.model_list=[]
            self.memory.model_readme=[]
            self.memory.merged_readme=None
            self.memory.remote_folder=""
            self.memory.ssh_config={}
            self.memory.github_config={}
            self.memory.model_url_list=[]
            self.memory.model_id_list=[]
            self.memory.reference_example=""
            self.memory.merged_reference_example=""
            self.memory.execution_result=[]
            self.memory.executed_command=""
            self.save_to_file()

    def save_to_file(self)->bool:
        try:
            data={
                "model_list":self.memory.model_list,
                "model_readme":self.memory.model_readme,
                "merged_readme":self.memory.merged_readme,
                "remote_folder":self.memory.remote_folder,
                "ssh_config":self.memory.ssh_config,
                "github_config":self.memory.github_config,
                "model_url_list":self.memory.model_url_list,
                "model_id_list":self.memory.model_id_list,
                "reference_example":self.memory.reference_example,
                "merged_reference_example":self.memory.merged_reference_example,
                "execution_result":self.memory.execution_result,
                "executed_command":self.memory.executed_command
            }
            with open(self.persist_path,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            return True
        except Exception as e:
            return False

    def memory_store(self,key:str,value:Any)->bool:
        try:
            self.load_from_file()
            if key=="model_list":
                self.memory.model_list=value
            elif key=="merged_readme":
                self.memory.merged_readme=value
            elif key=="model_readme":
                self.memory.model_readme=value
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
            elif key=="reference_example":
                self.memory.reference_example=value
            elif key=="merged_reference_example":
                self.memory.merged_reference_example=value
            elif key=="execution_result":
                self.memory.execution_result=value
            elif key=="executed_command":
                self.memory.executed_command=value
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
            elif key=="merged_readme":
                return self.memory.merged_readme
            elif key=="model_readme":
                return self.memory.model_readme
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
            elif key=="reference_example":
                return self.memory.reference_example
            elif key=="merged_reference_example":
                return self.memory.merged_reference_example
            elif key=="execution_result":
                return self.memory.execution_result
            elif key=="executed_command":
                return self.memory.executed_command
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
        # return ["model_list","merged_readme","model_readme","remote_folder","ssh_config","github_config","model_url_list","model_id_list","reference_example","merged_reference_example","execution_result"]
        memory=GlobalMemory()
        return memory.get_memory_keys()
    
    @tool("Get Memory Value Type")
    def get_memory_value_type():
        """"""
        memory=GlobalMemory()
        return memory.get_memory_value_types()