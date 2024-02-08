import os
from typing import Literal
from pydantic import BaseModel
from typing_extensions import Annotated

import chromadb
import autogen

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

# Reset agents
def reset_agents(agents):
    for agent in agents:
        agent.reset()

# Start chat function
def start_chat(agents, problem, llm_config):
    reset_agents(agents)
    groupchat = autogen.GroupChat(
        agents=agents, messages=[], max_round=20, 
        speaker_selection_method="auto",  allow_repeat_speaker=False)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    agents[0].initiate_chat(manager, problem=problem, n_results=1)

# Define the problem statement
PROBLEM = "What are the GDP figures for the USA and Germany? Additionally, determine which country has the higher GDP and output GDP in their respective national currencies. Output final answer of each sub questions as one final answer."

# Define agents
boss = autogen.RetrieveUserProxyAgent(
    name="Boss",
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "/content/drive/MyDrive/sample_doc/sample_1.pdf",
            "/content/drive/MyDrive/sample_doc/sample_2.pdf",
            "/content/drive/MyDrive/sample_doc/sample_3.pdf"
        ],
        "chunk_token_size": 500,
        "max_tokens":1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.,
)

currency_aid = autogen.AssistantAgent(
    name="currency_assistant",
    system_message="Suggest currency of given countries and convert it. For currency conversion tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config={"timeout": 60, "cache_seed": 42, "config_list": config_list, "temperature": 0},
)

planning_assistant = autogen.AssistantAgent(
    name="planning_assistant",
    system_message="Assistant who has extra planning power for solving difficult problems.",
    llm_config={"timeout": 600, "cache_seed": 42, "config_list": config_list},
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
@boss.register_for_execution()
@currency_aid.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"

# Start the chat
start_chat([boss, currency_aid, planning_assistant], PROBLEM, {"config_list": config_list})
