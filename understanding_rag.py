import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
import chromadb
import os

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

assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

# Accepted file formats for that can be stored in 
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

print("Accepted file formats for `docs_path`:")
print(TEXT_FORMATS)


# # 1. create an RetrieveAssistantAgent instance named "assistant"
# assistant = RetrieveAssistantAgent(
#     name="assistant",
#     system_message="You are a helpful assistant.",
#     llm_config={
#         "timeout": 600,
#         "cache_seed": 42,
#         "config_list": config_list,
#     },
# )

# # 2. create the RetrieveUserProxyAgent instance named "ragproxyagent"
# # By default, the human_input_mode is "ALWAYS", which means the agent will ask for human input at every step. We set it to "NEVER" here.
# # `docs_path` is the path to the docs directory. It can also be the path to a single file, or the url to a single file. By default,
# # it is set to None, which works only if the collection is already created.
# # `task` indicates the kind of task we're working on. In this example, it's a `code` task.
# # `chunk_token_size` is the chunk token size for the retrieve chat. By default, it is set to `max_tokens * 0.6`, here we set it to 2000.
# # `custom_text_types` is a list of file types to be processed. Default is `autogen.retrieve_utils.TEXT_FORMATS`.
# # This only applies to files under the directories in `docs_path`. Explictly included files and urls will be chunked regardless of their types.
# # In this example, we set it to ["mdx"] to only process markdown files. Since no mdx files are included in the `websit/docs`,
# # no files there will be processed. However, the explicitly included urls will still be processed.
# ragproxyagent = RetrieveUserProxyAgent(
#     name="ragproxyagent",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=3,
#     retrieve_config={
#         "task": "code",
#         "docs_path": [
#             "/content/drive/MyDrive/sample_doc/sample_1.pdf",
#             "/content/drive/MyDrive/sample_doc/sample_2.pdf",
#             "/content/drive/MyDrive/sample_doc/sample_3.pdf"
#         ],
#         "custom_text_types": ["mdx"],
#         "chunk_token_size": 1000,
#         "model": config_list[0]["model"],
#         "client": chromadb.PersistentClient(path="/tmp/chromadb"),
#         "embedding_model": "all-mpnet-base-v2",
#         "get_or_create": True,  # set to False if you don't want to reuse an existing collection, but you'll need to remove the collection manually
#     },
#     code_execution_config=False, # set to False if you don't want to execute the code
# )


# # reset the assistant. Always reset the assistant before starting a new conversation.
# assistant.reset()

# # given a problem, we use the ragproxyagent to generate a prompt to be sent to the assistant as the initial message.
# # the assistant receives the message and generates a response. The response will be sent back to the ragproxyagent for processing.
# # The conversation continues until the termination condition is met, in RetrieveChat, the termination condition when no human-in-loop is no code block detected.
# # With human-in-loop, the conversation will continue until the user says "exit".
# PROBLEM= "What are the GDP figures for the USA and Germany? Additionally, determine which country has the higher GDP and output GDP in their respective national currencies. Output final answer of each sub questions as one final answer."

# ragproxyagent.initiate_chat(assistant, problem=PROBLEM, 
#                             search_string="GDP", n_results=1)  # search_string is used as an extra filter for the embeddings search, in this case, we only want to search documents that contain "spark"
#.




llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

# autogen.ChatCompletion.start_logging()


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
)

boss_aid = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "/content/drive/MyDrive/sample_doc/sample_1.pdf",
            "/content/drive/MyDrive/sample_doc/sample_2.pdf",
            "/content/drive/MyDrive/sample_doc/sample_3.pdf"
        ],
        "custom_text_types": ["mdx"],
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "embedding_model": "all-mpnet-base-v2",
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection, but you'll need to remove the collection manually
    },
    code_execution_config=False, # set to False if you don't want to execute the code
)

coder = AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

pm = autogen.AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

retriever = autogen.AssistantAgent(
    name="retriever",
    is_termination_msg=termination_msg,
    system_message="You are a assistant who has extra content retrieval power for solving difficult problems. For retriver tasks, only use the functions you have been provided with. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)


PROBLEM = "What are the GDP figures for the USA and Germany? Output final answer of each sub questions as one final answer."

def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    pm.reset()
    reviewer.reset()
    retriever.reset()
    

_reset_agents()

# In this case, we will have multiple user proxy agents and we don't initiate the chat
# with RAG user proxy agent.
# In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
# it from other agents.
@boss.register_for_execution()
@retriever.register_for_llm(name="retrieve_content", description="retrieve content forquestion answering and return the execution result.")
def retrieve_content(message):
    n_results = 1
    boss_aid.n_results = n_results  # Set the number of results to be retrieved.
    # Check if we need to update the context.
    update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
    if (update_context_case1 or update_context_case2) and boss_aid.update_context:
        boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
        _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
    else:
        ret_msg = boss_aid.generate_init_message(message, n_results=n_results)
    return ret_msg if ret_msg else message

boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

llm_config = {
    "functions": [
        {
            "name": "retrieve_content",
            "description": "retrieve content for code generation and question answering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                    }
                },
                "required": ["message"],
            },
        },
    ],
    "config_list": config_list,
    "timeout": 60,
    "cache_seed": 42,
}

# for agent in [retriever]:
#     # update llm_config for assistant agents.
#     agent.llm_config.update(llm_config)
#     print(f"{agent.name}_tools_function: ,{agent.llm_config['functions']}")

# for agent in [boss]:
#     # register functions for all agents.
#     agent.register_function(
#         function_map={
#             "retrieve_content": retrieve_content,
#         }
#     )
#     print(f"{agent.name}_function_map: ,{agent.function_map}")
    


groupchat = autogen.GroupChat(
    agents=[boss, retriever],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",
    allow_repeat_speaker=False,
)

manager_llm_config = llm_config.copy()
manager_llm_config.pop("functions")
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

# Start chatting with the boss as this is the user proxy agent.
boss.initiate_chat(
    manager,
    message=PROBLEM,
)
