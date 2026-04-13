import json
import time

def create_step_callback(agent_name):
    def on_agent_step(step_output):
        step_data={
            "timestamp":time.time(),
            "agent":step_output.agent.role,
            "thought":step_output.thought,
            "action":step_output.action,
            "tool_output":step_output.tool_output,
            "final_output":step_output.final_output,
        }
        print(f"{agent_name}的流程")
        print(f"\n【步骤输出 | {step_output.agent.role}】")
        print(f"思考: {step_output.thought}")
        print(f"行动: {step_output.action}")
        print(f"结果: {step_output.final_output}")

        with open(f"/home/changrui/readme_generator/src/mid_res/readme_generator/{agent_name}.json","w",encoding="utf-8") as f:
            json.dump(step_data,f,ensure_ascii=False,indent=2)
    return on_agent_step