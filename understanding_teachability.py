import autogen
from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen import UserProxyAgent

config_list = autogen.config_list_from_json(
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


llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "cache_seed": None,  # Use an int to seed the response cache. Use None to disable caching.
}

teach_config={
    "verbosity": 0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    "reset_db": True,  # Set to True to start over with an empty database.
    "path_to_db_dir": "./tmp/notebook/teachable_agent_db",  # Path to the directory where the database will be stored.
    "recall_threshold": 1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
}

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x
    
teachable_agent = TeachableAgent(
    name="teachableagent",
    llm_config=llm_config,
    teach_config=teach_config)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
)


text = "What is the Vicuna model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)

text = "Vicuna is a 13B-parameter language model released by Meta."
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "What is the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "Orca is a 13B-parameter language model released by Microsoft. It outperforms Vicuna on most tasks."
user.initiate_chat(teachable_agent, message=text, clear_history=False)


#The following function needs to be called at the end of each chat, so that `TeachableAgent` can store what the user has taught it.
teachable_agent.learn_from_user_feedback()


text = "How does the Vicuna model compare to the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)