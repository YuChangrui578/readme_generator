from typing import Any,List,Optional
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field
import openai

class CustomChatOpenAI(BaseChatModel):
    base_url:str
    api_key:Optional[str]=None
    temperature:float=0.7
    model:str="unknown_model"

    @property
    def _llm_type(self)->str:
        return "custom_openai"
    
    def _generate(
        self,
        messages:List[BaseMessage],
        stop:Optional[List[str]]=None,
        **kwargs:Any
    )->ChatResult:
        client=openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "dummy"
        )

        msg_dicts=[{"role":msg.type,"content":msg.content} for msg in messages]
        response=client.chat.completions.create(
            messages=msg_dicts,
            stop=stop,
            **kwargs
        )
        content=response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(messages=BaseMessage(content=content,type="assistant"))]
        )