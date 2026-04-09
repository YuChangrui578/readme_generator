from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from readme_generator.tools.memory_tool import MemoryTool


@CrewBase
class ReadmeGeneratorCrew:
    agents_config="config/readme_generate_agents.yaml"
    tasks_config="config/readme_generate_tasks.yaml"
    # llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )

    @agent
    def readme_generator_agent(self)->Agent:
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        memory_get_key_tool=MemoryTool.get_memory_key
        return Agent(
            config=self.agents_config["readme_generator_agent"],
            llm=self.llm,
            tools=[memory_store_tool,memory_retrieve_tool,memory_get_key_tool],
            verbose=True,
            allow_delegation=True
        )
    
    @task
    def readme_generate(self)->Task:
        return Task(config=self.tasks_config["readme_generate_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )