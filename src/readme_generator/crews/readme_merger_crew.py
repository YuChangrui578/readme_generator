from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from readme_generator.tools.merge_content_tool import MergeContentTool
from readme_generator.tools.memory_tool import MemoryTool


@CrewBase
class ReadmeMergerCrew:
    agents_config="config/readme_merge_agents.yaml"
    tasks_config="config/readme_merge_tasks.yaml"
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )
    @agent
    def merge_readme_agent(self)->Agent:
        merge_readme_tool=MergeContentTool.merge_readme
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        memory_get_key_tool=MemoryTool.get_memory_key
        return Agent(
            config=self.agents_config["merge_readme_agent"],
            tools=[merge_readme_tool,memory_store_tool,memory_get_key_tool,memory_retrieve_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    @task
    def readme_merge(self)->Task:
        return Task(config=self.tasks_config["readme_merge_task"])

    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )