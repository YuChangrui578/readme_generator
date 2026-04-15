import json
import os
import requests
from github import Github,GithubException
from github.Repository import Repository
from github.Branch import Branch
from github.PullRequest import PullRequest
from github.ContentFile import ContentFile
from crewai import Agent,Task
# from langchain.tools import tool
from crewai.tools import tool
from typing import Dict,Optional,Tuple
import traceback
from .web_tool import backup_proxy_in_process,clear_proxy_in_process,restore_proxy_in_process
from .memory_tool import GlobalMemory

GITHUB_API="https://api.github.com"


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

    def create_pr(self,head_branch:str,base_branch:str,title:str,body:str)->PullRequest:
        return self.repo.create_pull(title=title,body=body,head=head_branch,base=base_branch)

    def update_pr(self,pr:PullRequest,title:str,body:str):
        pr.edit(title=title,body=body)

    def upsert_file(self,branch_name:str,path:str,content:str,message:str):
        content_bytes = content.encode("utf-8")
        try:
            existing = self.repo.get_contents(path, ref=branch_name)
            if isinstance(existing, list):
                existing = existing[0]
            self.repo.update_file(
                path=path,
                message=message,
                content=content_bytes,
                sha=existing.sha,
                branch=branch_name
            )
        except GithubException as e:
            if 404 in e.args:
                self.repo.create_file(
                    path=path,
                    message=message,
                    content=content_bytes,
                    branch=branch_name
                )
            else:
                raise e
            
    def create_empty_commit(self,branch_name):
        branch=self.get_branch(branch_name=branch_name)
        latest_commit = branch.commit
        tree = latest_commit.commit.tree

        new_commit=self.repo.create_git_commit(
            message="chore:empty commit to enable PR",
            tree=tree,
            parents=[latest_commit.commit]
        )
        ref=self.repo.get_git_ref(f"heads/{branch_name}")
        ref.edit(sha=new_commit.sha)

class PRPipeline:
    def __init__(self,token:str,repo_name:str,base_branch:str="main"):
        self.client=GitHubClient(token,repo_name)
        self.base_branch=base_branch

    def step1_check_resources(self,branch_name:str):
        branch=self.client.get_branch(branch_name=branch_name)
        if not branch:
            self.client.create_branch(branch_name=branch_name,base_branch=self.base_branch)
        existing_pr=self.client.get_open_pr(branch_name=branch_name)
        return existing_pr

    def step2_ensure_pr(self,branch_name:str,pr_title:str,pr_body:str,
                        existing_pr:Optional[PullRequest]):
        try:
            if existing_pr:
                self.client.update_pr(existing_pr,pr_title,pr_body)
                return existing_pr
            else:
                return self.client.create_pr(
                    head_branch=branch_name,
                    base_branch=self.base_branch,
                    title=pr_title,
                    body=pr_body
                )
        except GithubException as e:
            if "No commits between" in str(e):
                self.client.create_empty_commit(branch_name=branch_name)
                return self.client.create_pr(
                    head_branch=branch_name,
                    base_branch=self.base_branch,
                    title=pr_title,
                    body=pr_body
                )
            else:
                print(e)
                traceback.print_exc()

    def step3_commit_single_file(self, branch_name: str, path: str, content: str):
        msg = f"docs: update {path}"
        try:
            self.client.upsert_file(branch_name, path, content, msg)
        except Exception as e:
            traceback.print_exc()
            raise e


class GithubPRTool():
    
    @tool("upload_pr")
    def upload_pr_for_repo(github_token:str,repo:str,base_branch:str,branch_name:str,path:str,content:str,pr_title:str,pr_body:str)->Optional[PullRequest]:
        """Based on the relevant parameters required for the PR, submit the PR to the corresponding branch of the repository."""
        proxy_backup={
            "http_proxy":"http://proxy-dmz.intel.com:912",
            "https_proxy":"http://proxy-dmz.intel.com:912"
        }
        try:
            proxy_backup=restore_proxy_in_process(proxy_backup=proxy_backup)
            pipeline=PRPipeline(github_token,repo,base_branch)
            existing_pr=pipeline.step1_check_resources(branch_name)
            final_pr=pipeline.step2_ensure_pr(branch_name,pr_title,pr_body,existing_pr)
            pipeline.step3_commit_single_file(branch_name, path, content) 
            return final_pr
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            proxy_backup=clear_proxy_in_process(proxy_backup=proxy_backup)
        
    @tool("validate_pr")
    def validate_pr_exists_for_repo(github_token:str,repo:str,branch_name:str):
        """Verify whether the PR for the corresponding repository branch exists based on the relevant parameters required for the PR."""
        proxy_backup={
            "http_proxy":"http://proxy-dmz.intel.com:912",
            "https_proxy":"http://proxy-dmz.intel.com:912"
        }
        try:
            proxy_backup=restore_proxy_in_process(proxy_backup=proxy_backup)
            client=GitHubClient(github_token,repo)
            pr=client.get_open_pr(branch_name=branch_name)
            return pr is not None,pr
        except Exception as e:
            return False,None
        finally:
            proxy_backup=clear_proxy_in_process(proxy_backup=proxy_backup)

    @tool("get_github_config")
    def get_github_config():
        """Retrieve GITHUB_CONFIG from GLOBAL_MEMORY.
        Returns: dictionary containing github_token, repo_owner, repo_name, base_branch, head_branch, pr_title, pr_description, commit_message, path."""
        memory=GlobalMemory()
        return memory.memory_retrieve("github_config") or {}
    
    @tool("get_merged_readme")
    def get_merged_readme():
        """Retrieve merged_readme from GLOBAL_MEMORY.
        Returns: merged README content string."""
        memory=GlobalMemory()
        return memory.memory_retrieve("merged_readme") or ""
    
    @tool("memory_store_pr_info")
    def memory_store_pr_info(pr_num,url,status):
        """Store PR information into GLOBAL_MEMORY.
        Inputs: PR number, PR URL, PR status
        Returns: success message."""
        memory=GlobalMemory()
        memory.memory_store("pr_info",{
            "number":pr_num,
            "url":url,
            "status":status
        })
        
        
            
            
    
