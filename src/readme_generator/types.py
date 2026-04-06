from pydantic import BaseModel

class Github(BaseModel):
    github_token:str
    repo:str
    base_branch:str
    branch_name:str


class Readme_generator(BaseModel):
    model_name:str
    model_url:str

class Run_code(BaseModel):
    user_name:str
    host_id:str
    password:str
    remote_folder:str

