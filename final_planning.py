import os
import chromadb
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from IPython import get_ipython


config_file_or_env = "/content/drive/MyDrive/OAI_CONFIG_LIST"
config_list = autogen.config_list_from_json(
    config_file_or_env,
    filter_dict={
        "model": {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "chatgpt-35-turbo-0301",
            "gpt-35-turbo-v0301",
            "gpt-4", "gpt4", "gpt-4-32k"
        },
    },
)

llm_config = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])


# Define agents
pm = autogen.AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

writer = autogen.AssistantAgent(
    name="writer",
    is_termination_msg=termination_msg,
    llm_config={"config_list": config_list},
    # the default system message of the AssistantAgent is overwritten here
    system_message="You are a movie writer. Your responsible for crafting compelling narratives and dialogue for given movies description. Reply `TERMINATE` in the end when everything is done."
)

coder = autogen.AssistantAgent(
    name="coder",
    is_termination_msg=termination_msg,
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

planner = autogen.AssistantAgent(
    name="planner",
    llm_config={"config_list": config_list},
    # the default system message of the AssistantAgent is overwritten here
    system_message="You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning, convert it to a step that can be implemented by writing code. For example, browsing the web can be implemented by writing code that reads and prints the content of a web page. Finally, inspect the execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze the error and suggest a fix.",
)

planner_user = autogen.UserProxyAgent(
    name="planner_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
)


def ask_planner(message):
    planner_user.initiate_chat(planner, message=message)
    # return the last message received from the planner
    return planner_user.last_message()["content"]

planning_aid = autogen.AssistantAgent(
    name="planning_assistant",
    llm_config={
        "temperature": 0,
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
        "functions": [
            {
                "name": "ask_planner",
                "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
                        },
                    },
                    "required": ["message"],
                },
            },       
        ],
    },
)
CurrencySymbol = Literal["USD", "EUR"]


def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")

# Define currency calculator function
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"


currency_aid = autogen.AssistantAgent(
    name="currency_assistant",
    system_message="Suggest currency of given countries and convert it. For currency conversion tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config={"timeout": 60, 
                "cache_seed": 42, 
                "config_list": config_list, 
                "temperature": 0,
                "functions":[
                    {
                        'description': 'Currency exchange calculator.', 
                        'name': 'currency_calculator', 
                        'parameters': {
                            'type': 'object', 
                            'properties': {
                                'base_amount': {'type': 'number', 'description': 'Amount of currency in base_currency'}, 
                                'base_currency': {'enum': ['USD', 'EUR'], 'type': 'string', 'default': 'USD', 'description': 'Base currency'}, 
                                'quote_currency': {'enum': ['USD', 'EUR'], 'type': 'string', 'default': 'EUR', 'description': 'Quote currency'}}, 
                            'required': ['base_amount']}
                    }
                ]
            },
    )

boss_aid = RetrieveUserProxyAgent(
    name="ragproxyagent",
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER", # Disable human input for boss_aid since it only retrieves content.
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

retriever_aid = autogen.AssistantAgent(
    name="retriever_assistant",
    is_termination_msg=termination_msg,
    system_message="You are a assistant who has extra content retrieval power for solving difficult problems. For retriver tasks, only use the functions you have been provided with. Reply `TERMINATE` in the end when everything is done.",
    llm_config= {
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
},
)

boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config=False,  # we don't want to execute code in this case.
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    function_map={"retrieve_content": retrieve_content, "currency_calculator": currency_calculator, "ask_planner":ask_planner}
)

print("********printing agent tool********")
print(f"{currency_aid.name}_tools_function: ,{currency_aid.llm_config['functions']}")
print(f"{retriever_aid.name}_tools_function: ,{retriever_aid.llm_config['functions']}")
print(f"{planning_aid.name}_tools_function: ,{planning_aid.llm_config['functions']}")
# print(f"{coder.name}_tools_function: ,{coder.llm_config['tools'][0]['function']}")
print(f"{boss.name}_function_map: ,{boss.function_map}")

# Reset agents
def reset_agents(agents):
    for agent in agents:
        agent.reset()

# Start chat function
def start_chat(agents, problem, llm_config):
    reset_agents(agents)
    groupchat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=20,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )

    manager_llm_config = llm_config.copy()
    # manager_llm_config.pop("functions")
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    agents[0].initiate_chat(
        manager,
        message=problem,
    )

# Define the problem statement
PROBLEM = "What are the GDP figures for the USA and Germany? Additionally, determine which country has the higher GDP and output GDP in their respective national currencies. Output final answer of each sub questions as one final answer."


# Start the chat
start_chat([boss, planning_aid, retriever_aid, currency_aid, pm, coder, writer], PROBLEM, {"config_list": config_list})
