from typing import Any,Optional,List,Dict
from crewai.tools import tool
import traceback
import json
import os

class GlobalMemory:
    def __init__(
        self,
        persist_path:str="global_memory.json"
    ):
        self.persist_path=persist_path
        self.load_from_file()

    def load_from_file(self)->None:
        if os.path.exists(self.persist_path):
            with open(self.persist_path,"r",encoding="utf-8") as f:
                data=json.load(f)
                self.model_list=data.get("model_list",[])
                self.models=data.get("models",{})
                self.merged_readme=data.get("merged_readme","")
                self.remote_folder=data.get("remote_folder","")
                self.ssh_config=data.get("ssh_config",{})
                self.github_config=data.get("github_config",{})
                # 需要处理
        else:
            self.model_list:List[str]=[]
            self.models:Dict[str,Dict[str,Any]]={}
            self.merged_readme:Optional[str]=None
            self.remote_folder:str=""
            self.ssh_config:Dict[str,Any]={}
            self.github_config:Dict[str,Any]={}
            self.save_to_file()

    def save_to_file(self)->bool:
        try:
            data={
                "model_list":self.model_list,
                "models":self.models,
                "merged_readme":self.merged_readme
            }
            with open(self.persist_path,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            return True
        except Exception as e:
            return False

    def memory_store(self,key:str,value:Any,sub_key:Optional[str]=None)->bool:
        try:
            if key=="model_list":
                self.model_list=value
                self.models={
                    name:info for name,info in self.models.items()
                    if name in self.model_list
                }
            elif key=="merged_readme":
                self.merged_readme=value
            elif key in self.models or key in self.model_list:
                if key not in self.models:
                    self.models[key]={}
                self.models[key][sub_key]=value
            self.save_to_file()
            return True
        except Exception as e:
            print(e)
            traceback.print_exc()
            return False
        
    def memory_retrieve(self,key:str,sub_key:Optional[str]=None,default:Any=None)->Any:
        if key=="model_list":
            return self.model_list.copy()
        if key=="merged_readme":
            return self.merged_readme
        model_info=self.models.get(key,{})
        if sub_key is None:
            return model_info
        return model_info.get(sub_key,default)

class MemoryTool:
    @tool("Store Memory")
    def store_memory(key:str,value:Any,sub_key:Optional[str]=None):
        """Used to persistently store key model information, configuration data, and intermediate results generated during task execution into global memory and synchronously save them to a local file to ensure data integrity. Supports storing the model list, detailed information of individual models (model URL, README content, shell commands, execution results, etc.), and merged README documents. It strictly verifies whether a model belongs to the valid list during storage to maintain consistency and standardization of global data. Each storage operation is automatically saved to disk, allowing all subsequent agents to access and reuse data across tasks and workflows."""
        memory=GlobalMemory()
        flag=memory.memory_store(key=key,value=value,sub_key=sub_key)
        return flag
    
    @tool("Retrieve Memory")
    def retrieve_memory(key:str,sub_key:Optional[str]=None,default:Any=None):
        """Used to retrieve various stored data from the global persistent memory. Supports fetching the valid model list, detailed information of a specified model (model URL, README, execution commands, etc.), and the final merged README document. Data is loaded from a local file, ensuring consistent and complete content across tasks and agents. Returns a default value if the requested data does not exist, maintaining stable and uninterrupted task execution."""
        memory=GlobalMemory()
        info=memory.memory_retrieve(key=key,sub_key=sub_key,default=default)
        return info