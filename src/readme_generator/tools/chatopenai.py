import openai
import copy
from openai import OpenAI,AsyncOpenAI
import traceback

class LLM_Callable:
    def __init__(self,base_url,api_key,model_name):
        self.base_url=base_url
        self.api_key=api_key
        self.model_name=model_name
        self.client=openai.Client(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
    def invoke(self,inputs):
        try:
            response=self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role":"user","content":inputs}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ""


