import autogen
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from autogen import ConversableAgent, UserProxyAgent

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

print(config_list[0]["model"])

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
    
# Start by instantiating any agent that inherits from ConversableAgent.
teachable_agent = ConversableAgent(
    name="teachable_agent",  # The name is flexible, but should not contain spaces to work in group chat.
    llm_config={"config_list": config_list, "timeout": 120, "cache_seed": None},  # Disable caching.
)

# Instantiate the Teachability capability. Its parameters are all optional.
teachability = Teachability(
    verbosity=2,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=True,
    path_to_db_dir="./tmp/notebook/teachability_db",
    recall_threshold=1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

# Now add the Teachability capability to the agent.
teachability.add_to_agent(teachable_agent)

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


# Instantiate a UserProxyAgent to represent the user. But in this notebook, all user input will be simulated.
user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

text = "What is the Vicuna model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)

text = "Vicuna is a 13B-parameter language model released by Meta."
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "What is the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "Orca is a 13B-parameter language model developed by Microsoft. It outperforms Vicuna on most tasks."
user.initiate_chat(teachable_agent, message=text, clear_history=False)

text = "How does the Vicuna model compare to the Orca model?"
user.initiate_chat(teachable_agent, message=text, clear_history=True)