from autogen.agentchat.contrib.agent_builder import AgentBuilder
import autogen

config_file_or_env = "/content/drive/MyDrive/OAI_CONFIG_LIST"  # modify path
llm_config = {"temperature": 0}
config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-3.5-turbo-0301","chatgpt-35-turbo-0301","gpt-35-turbo-v0301",]})

builder = AgentBuilder(config_file_or_env=config_file_or_env, builder_model='gpt-3.5-turbo', agent_model='gpt-3.5-turbo')
building_task = "Find a paper on arxiv by programming, and analyze its application in some domain. For example, find a recent paper about gpt-4 on arxiv and find its potential applications in software."

agent_list, agent_configs = builder.build(building_task, llm_config)
print()

def start_task(execution_task: str, agent_list: list):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

start_task(
    execution_task="Find a recent paper about LLM agents on arxiv and find its potential applications in software.",
    agent_list=agent_list
)