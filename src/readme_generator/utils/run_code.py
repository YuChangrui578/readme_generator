import os
import time
<<<<<<< HEAD

from typing import List

from readme_generator.type import Run_code

def check_run(run_inform:Run_code):
    pass
=======
import re
from typing import List,Dict

from readme_generator.type import Run_code

class CodeParser:
    @staticmethod
    def extract_shell_commands(markdown_content:str)->List[Dict[str,str]]:
        pattern = r"```(\w+)?\n(.*?)```"
        matches=re.findall(pattern,markdown_content,re.DOTALL)
        commands=[]
        shell_languages=["bash","sh","shell","cmd","console","zsh"]
        for lang,code in matches:
            lang_lower=(lang or "").lower()
            if lang_lower in shell_languages:
                clean_code = re.sub(r'^[\$\>]\s*', '', code, flags=re.MULTILINE)\
                if clean_code.strip() and not clean_code.strip().startwith("#"):
                    commands.append({
                        "language":lang_lower,
                        "code":clean_code.strip()
                    })
        return commands
>>>>>>> d671c9d (init)

def wait_next_run(state):
    print("## Waiting for 180 seconds")
    time.sleep(180)
    return state