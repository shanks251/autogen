import autogen
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.agentchat.contrib.compressible_agent import CompressibleAgent

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

###example 1

# 1. replace AssistantAgent with CompressibleAgent
assistant = CompressibleAgent(
    name="assistant", 
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
    compress_config={
        "mode": "COMPRESS",
        "trigger_count": 600, # set this to a large number for less frequent compression
        "verbose": True, # to allow printing of compression information: contex before and after compression
        "leave_last_n": 2,
    }
)

# 2. create the MathUserProxyAgent instance named "mathproxyagent"
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent", 
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
    max_consecutive_auto_reply=5,
)
math_problem = "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."
mathproxyagent.initiate_chat(assistant, problem=math_problem)



###example 2

def constrain_num_messages(messages):
    """Constrain the number of messages to 3.
    
    This is an example of a customized compression function.

    Returns:
        bool: whether the compression is successful.
        list: the compressed messages.
    """
    if len(messages) <= 3:
        # do nothing
        return False, None
    
    # save the first and last two messages
    return True, messages[:1] + messages[-2:]

# create a CompressibleAgent instance named "assistant"
assistant = CompressibleAgent(
    name="assistant",
    llm_config={
        "timeout": 600,
        "cache_seed": 43,
        "config_list": config_list,
    },
    compress_config={
        "mode": "CUSTOMIZED",
        "compress_function": constrain_num_messages,  # this is required for customized compression
        "trigger_count": 1000, 
    },
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE") or x.get("content", "").rstrip().endswith("TERMINATE."),
    code_execution_config={"work_dir": "web"},
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
)

user_proxy.initiate_chat(
    assistant,
    message="""Show me the YTD gain of 10 largest technology companies as of today.""",
)
