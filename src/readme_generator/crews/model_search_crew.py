from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from tools.model_search_tool import ModelSearchTool
from tools.memory_tool import MemoryTool
from tools.chatopenai import CustomChatOpenAI
from langchain_openai import ChatOpenAI

@CrewBase
class ModelSearchCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    # llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )

    @agent
    def model_search_agent(self)->Agent:
        model_search_tool=ModelSearchTool.huggingface_model_search_url
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        return Agent(
            config=self.agents_config["model_search_agent"],
            tools=[model_search_tool,memory_store_tool,memory_retrieve_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )
    
    @task
    def model_search(self)->Task:
        import pdb;pdb.set_trace()
        return Task(config=self.tasks_config["model_search_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )