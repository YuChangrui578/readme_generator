from crewai.tools import tool
from typing import List,Optional

class GenerateReadmeTool():

    @tool("Get Reference Example")
    def get_reference_example(github_url:List[str],origin_reference_example_list:List[str])->List[str]:
        """根据github_url中每个元素是否为空来选择其reference_example,返回的是一个List,其每个元素都对应的model_id_list对应位置代表模型的reference_example"""
        reference_example_list=[]
        for u in github_url:
            if len(u)>0:
                reference_example_list.append(origin_reference_example_list[1])
            else:
                reference_example_list.append(origin_reference_example_list[0])
        return reference_example_list
        
