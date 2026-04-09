from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from readme_generator.tools.memory_tool import MemoryTool

@CrewBase
class InputParserCrew:
    agents_config="config/input_parser_agents.yaml"
    tasks_config="config/input_parser_tasks.yaml"

    llm=LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty"
    )

    @agent
    def input_parser_agent(self)->Agent:
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        memory_get_key_tool=MemoryTool.get_memory_key

        return Agent(
            config=self.agents_config["input_parser_agent"],
            tools=[memory_get_key_tool,memory_retrieve_tool,memory_store_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    @task
    def input_parse(self)->Task:
        return Task(config=self.tasks_config["input_parse_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )