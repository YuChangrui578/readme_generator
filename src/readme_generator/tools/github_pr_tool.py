import json
import os
import requests
from github import Github,GithubException
from github.Repository import Repository
from github.Branch import Branch
from github.PullRequest import PullRequest
from github.ContentFile import ContentFile
from crewai import Agent,Task
from langchain.tool import tool
from typing import Dict,Optional,Tuple
import traceback

GITHUB_API="https://api.github.com"

class GithubClient:
    def __init__(self,token:str,repo_name:str,base_branch:str="main"):
        self.client=Github(token)
        self.repo:Repository=self.client.get_repo(repo_name)
        self.base_branch=base_branch

    def get_or_create_branch(self,new_branch_name:str)->Branch:
        try:
            branch=self.repo.get_branch(new_branch_name)
            return branch
        except GithubException as e:
            source_branch=self.repo.get_branch(self.base_branch)
            self.repo.create_git_ref(ref=f"refs/heads/{new_branch_name}",sha=source_branch.commit.sha)
            return self.repo.get_branch(new_branch_name)
        
    def upsert_file(self,branch_name:str,file_path:str,content:str,commit_message:str):
        content_bytes=content.encode("utf-8")
        try:
            existing_file=self.repo.get_contents(file_path,ref=branch_name)
            if isinstance(existing_file,list):
                existing_file=existing_file[0]
            self.repo.update_file(
                path=file_path,
                message=commit_message,
                content=content_bytes,
                sha=existing_file.sha,
                branch=branch_name
            )
        except GithubException as e:
            if e.status==404:
                self.repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=content_bytes,
                    branch=branch_name
                )
            else:
                raise e
    
    def create_or_update_pr(self,branch_name:str,pr_titles:str,pr_body:str):
        open_pulls=self.repo.get_pulls(state="open",head=branch_name)
        if open_pulls.totalcount>0:
            pr=open_pulls[0]
            if pr.body!=pr_body:
                pr.edit(body=pr_body)
            return pr
    
    def run(self,text_content:str,branch_name:str,file_path:str="README.md"):
        self.get_or_create_branch(branch_name)

        commit_msg=f"docs:update {file_path} via automation"
        self.upsert_file(branch_name,file_path,text_content,commit_msg)

        pr_title=f"Update {file_path} from automation"
        pr_body=f"Automated update for `{file_path}`.\n\nGenerated content:\n```\n{text_content[:200]}...\n```"
        self.create_or_update_pr(branch_name, pr_title, pr_body)

class GitHubClient:
    def __init__(self,token:str,repo_name:str):
        self.g=Github(token)
        self.repo:Repository=self.g.get_repo(repo_name)

    def get_branch(self,branch_name:str)->Optional[Branch]:
        try:
            return self.repo.get_branch(branch_name)
        except GithubException:
            return None 
    
    def create_branch(self,branch_name:str,base_branch:str)->Branch:
        source=self.repo.get_branch(base_branch)
        self.repo.create_git_ref(ref=f"refs/heads/{branch_name}",sha=source.commit.sha)
        return self.repo.get_branch(branch_name)

    def get_open_pr(self,branch_name:str)->Optional[PullRequest]:
        pulls=self.repo.get_pulls(state="open",head=branch_name)
        return pulls[0] if pulls.totalCount>0 else None

    def create_pr(self,branch_name:str,base_branch:str,title:str,body:str)->PullRequest:
        return self.repo.create_pull(title=title,body=body,head=branch_name,base=base_branch)

    def update_pr(self,pr:PullRequest,title:str,body:str):
        pr.edit(title=title,body=body)

    def upsert_file(self,branch_name:str,path:str,content:str,message:str):
        content_bytes=content.encode("utf-8")
        try:
            existing=self.repo.get_contents(path,ref=branch_name)
            if isinstance(existing,list):
                existing=existing[0]
                self.repo.update_file(path=path,message=message,content=content_bytes,sha=existing.sha,branch=branch_name)
            except GithubException as e:
                raise e

class PRPipeline:
    def __init__(self,token:str,repo_name:str,base_branch:str="main"):
        self.client=GithubClient(token,repo_name)
        self.base_branch=base_branch

    def step1_check_resources(self,branch_name:str)->Tuple[Branch,bool,Optional[PullRequest]]:
        print(f"[Step 1]检查资源状态:分支'{branch_name}'...")
        branch=self.client.get_branch(branch_name)
        is_new_branch=False
        if not branch:
            print(f"分支不存在,准备创建...")
            branch=self.client.create_branch(branch_name,self.base_branch)
            is_new_branch=True  
            print(f"分支'{branch_name}'创建成功(基于{self.base_branch})")
        else:
            print(f"分支'{branch_name}'已存在")
        existing_pr=self.client.get_open_pr(branch_name)
        if existing_pr:
            print(f"发现开放PR #{existing_pr.number}:{existing_pr.title}")
        else:
            print("未发现该分支的开放PR")
        return branch,is_new_branch,existing_pr

    def step2_ensure_pr(self,branch_name:str,pr_title:str,pr_body:str,
                        existing_pr:Optional[PullRequest])->PullRequest:
        print(f"[Step 2]处理PR元数据")
        if existing_pr:
            print(f"更新现有PR #{existing_pr.number}信息...")
            if existing_pr.title!=pr_title or existing_pr.body!=pr_body:
                self.client.update_pr(existing_pr,pr_title,pr_body)
            else:
                print(f"PR信息无变化,跳过更新")
            return existing_pr
        print(f"创建新PR:'{pr_title}'...")
        new_pr=self.client.create_pr(branch_name,self.base_branch,pr_title,pr_body)
        print(f"PR #{new_pr.number}创建成功:{new_pr.html_url}")
        return new_pr

    def step3_commit_context(self,branch_name:str,context:Dict[str,str],commit_prefix:str="docs"):
        print(f"[Step 3]根据Context提交文件到'{branch_name}'...")
        if not context:
            print("Context为空,跳过提交")
            return 
        count=0
        for path,content in context.items():
            msg=f"{commit_prefix}:update {path} via pipeline"
            try:
                self.client.upsert_file(branch_name,path,content,msg)
                print(f"已处理:{path}")
                count+=1
            except Exception as e:
                print(f"失败:{path}-{e}")
                raise e
        print(f"成功提交{count}个文件")


class GithubPRTool():
    
    @tool("根据PR所需的相关参数,上传对应repo的branch的pr")
    def upload_pr_for_repo(github_token:str,repo:str,base_branch:str,branch_name:str,context:Dict[str,str],pr_title:str,pr_body:str)->Optional[PullRequest]:
        try:
            pipeline=PRPipeline(github_token,repo_name,base_branch)
            branch,is_new,existing_pr=self.pipeline.step1_check_resources(branch_name)
            final_pr=self.pipeline.step2_ensure_pr(branch_name,pr_title,pr_body,existing_pr)
            return final_pr
        except Exception as e:
            traceback.print_exc()
            raise e
        
    @tool("根据PR所需的相关参数,验证对应repo的branch的pr是否存在")
    def validate_pr_exists_for_repo(github_token:str,repo:str,base_branch:str,branch_name:str,state:str="open"):
        try:
            pipeline=PRPipeline(github_token,repo_name,base_branch)
            client=pipeline.client
            existing_pr=client.get_open_pr(branch_name)
            if state=="open":
                if existing_pr:
                    print(f"[需求 2]存在开放的PR:#{existing_pr.number}")
                    return True,existing_pr
                else:
                    print(f"[需求 2]不存在开放的PR")
                    return False,None
            else:
                pulls=self.client.repo.get_pulls(state=state,head=branch_name)
                if pulls.totalCount>0:
                    pr=pulls[0]
                    print(f"[需求 2]存在状态为'{state}'的PR:#{pr.number}")
                    return True,pr
                return False,None
        except Exception as e:
            print(f"[需求 2]验证出错:{e}")
            return False,None
            

    @tool("根据PR所需的相关参数，创建新的对应repo的branch的pr")
    def create_new_pr_for_repo(github_token:str,repo:str,base_branch:str,branch_name:str,pr_title:str,pr_body:str,force:bool=False)->Optional[PullRequest]:
        print(f"[需求 3]尝试创建新 PR:分支 {branch_name}")
        try:
            pipeline=PRPipeline(github_token,repo_name,base_branch)
            client=pipeline.client
            new_pr=client.create_pr(
                branch_name=branch_name,
                base_branch=pipeline.base_branch,
                title=pr_title,
                body=pr_body
            )
            print(f"[需求 3]成功创建新PR:#{new_pr.number}")
            return new_pr
        except Exception as e:
            print(f"[需求 3]创建失败:{e}")
            return None
            
    
