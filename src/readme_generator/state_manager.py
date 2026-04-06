from typing import List,Dict

class ModelState:
    def __init__(self,model_name:str,url:str,readme_segment:str,test_log_segment:str):
        self.model_name=model_name
        self.url=url
        self.readme_segment=readme_segment
        self.test_log_segment=test_log_segment

class GlobalState:
    _instance=None
    _states:List[ModelState]=[]

    def __new__(cls):
        if cls._instance is None:
            cls._instance=super(GlobalState,cls).__new__(cls)
        return cls._instance

    def add_result(self,state:ModelState):
        self._states.appned(state)
        print(f"Collected result for {state.model_name}.Total collected:{len(self._states)}")

    def get_all_states(self)->List[ModelState]:
        return self._states
    
    def clear(self):
        self._states=[]