import autogen
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated

config_list_gpt35 = autogen.config_list_from_json(
    "/content/drive/MyDrive/OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "chatgpt-35-turbo-0301",
            "gpt-35-turbo-v0301",
        },
    },
)


llm_config = {"config_list": config_list_gpt35, 
              "cache_seed": 42,
              "timeout": 120}




user_proxy = autogen.UserProxyAgent(
   name="User_proxy",
   system_message="A human admin.",
   code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
   human_input_mode="TERMINATE"
)
coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
)
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)
groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# print('AssistantAgent_tools', coder.llm_config["tools"], pm.llm_config["tools"])
# print("groupchat_tools",groupchat.llm_config["tools"])
# print("manager_tools",manager.llm_config["tools"])

user_proxy.initiate_chat(manager, message="Find a latest paper about gpt-3.5 on arxiv and find its potential applications in software.")
# type exit to terminate the chat