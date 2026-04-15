# import yagmail
# import traceback
# from datetime import datetime
# from langchain.tools import tool

# def send_model_test_failure_email(
#     model_id: str,
#     remote_folder: str,
#     error_detail: str,
#     fix_attempts: int = 3
# ):
#     """
#     Send email alert when remote model test fails AND self-fix failed multiple times.
#     Inputs:
#         - model_id: 模型ID (e.g., RedHatAI/Llama-3.2-3B-quantized.w8a8)
#         - remote_folder: 远程模型根目录
#         - error_detail: 完整错误信息/堆栈/日志
#         - fix_attempts: 已尝试自修复次数
#     Output: Send email to administrator and return send status.
#     """
#     try:
#         # ===================== 邮箱配置（你可以改成从配置/内存读取）=====================
#         smtp_user = "775499403@qq.com"       # 发件邮箱
#         smtp_password = "avqekgmfkpzzbegf"         # 授权码（不是邮箱密码）
#         smtp_host = "smtp.qq.com"               # SMTP 地址
#         receiver_list = ["changrui.yu@intel.com"]    # 接收人（可多个）
#         smtp_port = 465

#         now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         subject = f"【模型执行失败】{model_id} 远程测试失败（修复{fix_attempts}次无效）"

#         content = f"""
#         <h3>模型远程执行失败通知</h3>
#         <p><strong>发生时间：</strong> {now}</p>
#         <p><strong>模型ID：</strong> {model_id}</p>
#         <p><strong>远程目录：</strong> {remote_folder}</p>
#         <p><strong>修复尝试：</strong> {fix_attempts} 次均失败</p>
#         <br>
#         <h4>错误日志：</h4>
#         <pre style="background:#f5f5f5;padding:12px;border-radius:6px;">{error_detail}</pre>
#         <p>请人工检查模型路径、权限、远程环境配置。</p>
#         """
#         yag = yagmail.SMTP(
#             user=smtp_user,
#             password=smtp_password,
#             host=smtp_host,
#             port=smtp_port,
#             smtp_ssl=True
#         )
#         yag.send(to=receiver_list, subject=subject, contents=content)
#         return f"✅ 邮件已发送至：{receiver_list}"

#     except Exception as e:
#         return f"❌ 邮件发送失败：{str(e)}\n{traceback.format_exc()}"
    
# model_id="1"
# remote_folder="1"
# error_detail="1"
# result=send_model_test_failure_email(
#     model_id=model_id,
#     remote_folder=remote_folder,
#     error_detail=error_detail
# )
# import pdb;pdb.set_trace()
from tools.memory_tool import GlobalMemory
from tools.remote_exec_tool import RemoteGeneralExecutor

def check_remote_model_exists(model_id:str,remote_folder:str):
    """Check if the model exists in remote server's models folder.
    Inputs: model_id, remote_folder
    Returns: True if model exists, False otherwise."""
    from shlex import quote  # 安全转义路径，防止特殊字符/命令注入

    memory = GlobalMemory()
    ssh = memory.memory_retrieve("ssh_config")

    # 拼接完整模型路径（自动支持嵌套 model_id）
    model_full_path = f"{remote_folder}/models/{model_id}"
    # 安全转义，处理空格、/、.、-、: 等所有特殊字符
    safe_path = quote(model_full_path)

    # 安全判断远程目录是否存在（标准、兼容所有 Linux）
    cmd = f"test -d {safe_path} && echo EXISTS || echo NOT_EXISTS"

    executor = RemoteGeneralExecutor(
        host=ssh["hostname"],
        user_name=ssh["user_name"],
        password=ssh["password"],
        port=22
    )

    try:
        executor.connect()
        results = executor.execute_commands(
            command_list=[cmd],
            remote_folder=remote_folder
        )
        # 判断结果（兼容多行输出、空白字符）
        # 新代码（正确）
        output = results[0]["output"].strip()
        return "EXISTS" in output

    finally:
        # 确保无论是否异常，都断开连接
        executor.disconnect()

result=check_remote_model_exists(model_id="RedHatAI/Llama-3.2-3B-quantized.w8a8",remote_folder="/home/sdp/changrui")
print(result)