from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task

from readme_generator.tools.model_search import ModelSearchTool
from readme_generator.tools.github_pr import GithubPRTool
from readme_generator.tools.model_search_tool import ModelSearchTool
from readme_generator.tools.github_pr_tool import GithubPRTool
from readme_generator.tools.memory_store_tool import MemoryStoreTool
from readme_generator.tools.memory_retrieve_tool import MemoryRetrieveTool
from readme_generator.tools.merge_content_tool import MergeContentTool
from readme_generator.tools.remote_exec_tool import RemoteExecutionTool
from langchain_openai import ChatOpenAI

@CrewBase
class GithubCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    llm=ChatOpenAI(model="gpt-4o")

    @agent
    def github_agent(self)->Agent:
        github_tool=GithubPRTool()
        return Agent(
            config=self.agents_config["github_agent"],
            tools=[github_tool],
            llm=self.llm,
            verbose=True
        )

    @agent
    def readme_generator_agent(self)->Agent:
        model_search_tool=ModelSearchTool()
        return Agent(
            config=self.agents_config["readme_generator_agent"],
            llm=self.llm,
            tools=[model_search_tool],
            verbose=True,
            allow_delegation=True
        )
    
    @agent 
    def model_search_agent(self)->Agent:
        model_search_tool=ModelSearchTool()
        return Agent(
            config=self.agents_config["model_search_agent"],
            tools=[model_search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    @agent
    def merge_readme_agent(self)->Agent:
        merge_readme_tool=MergeContentTool()
        retrieve_tool=MemoryRetrieveTool()
        return Agent(
            config=self.agents_config["merge_readme_agent"],
            tools=[merge_readme_tool,retrieve_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

    @agent 
    def remote_execution_agent(self)->Agent:
        remote_execution_tool=RemoteExecutionTool()
        return Agent(
            config=self.agents_config["remote_execution_agent"],
            tools=[remote_execution_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

    @agent
    def memory_store_agent(self)->Agent:
        store_tool=MemoryStoreTool()
        return Agent(
            config=self.agents_config["memory_store_agent"],
            tools=[store_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )

    @task 
    def github_pr(self)->Task:
        return Task(config=self.tasks_config["github_pr"])
    
    @task
    def readme_generate(self)->Task:
        return Task(config=self.tasks_config["readme_generate"])
    
    @task
    def model_search(self)->Task:
        return Task(config=self.tasks_config["model_search"])

    @task
    def remote_execution(self)->Task:
        return Task(config=self.tasks_config["remote_execution"])

    @task
    def readme_merge(self)->Task:
        return Task(config=self.tasks_config["readme_merge"])

    @task 
    def memory_store(self)->Task:
        return Task(config=self.tasks_config["memory_store"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )