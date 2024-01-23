import os
import chromadb
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated


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

print("LLM models: ", [config_list[i]["model"] for i in range(len(config_list))])


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


PROBLEM = "What are the GDP figures for the USA and Germany? Additionally, determine which country has the higher GDP and output GDP in their respective national currencies. Output final answer of each sub questions as one final answer."


boss = RetrieveUserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
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
        # "max_tokens":1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

currency_aid = autogen.AssistantAgent(
    name="currency_assistant",
    system_message="Suggest currency of given countries and convert it. For currency conversion tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
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


@boss.register_for_execution()
@currency_aid.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"


# print("********printing agent tool********")
# print(f"{currency_aid.name}_tools_function: ,{currency_aid.llm_config['tools'][0]['function']}")
# print("********printing agent tool done********")


def _reset_agents():
    boss.reset()
    currency_aid.reset()


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss, currency_aid], messages=[], max_round=20, 
        speaker_selection_method="auto",  allow_repeat_speaker=False)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        problem=PROBLEM,
        n_results=1,
    )
  
rag_chat()
