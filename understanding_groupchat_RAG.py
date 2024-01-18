import os
import chromadb
import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated

config_list = autogen.config_list_from_json(
    "/content/drive/MyDrive/OAI_CONFIG_LIST",
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


boss = autogen.UserProxyAgent(
    name="Boss",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config=True,
    default_auto_reply="Reply `TERMINATE` if the task is done.",
)

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [
            "/content/drive/MyDrive/sample_doc/List of highest-grossing Indian films - Wikipedia.pdf",
            "/content/drive/MyDrive/sample_doc/SaaS_Product_AutoGen Agent_medium.pdf",
            "/content/drive/MyDrive/sample_doc/World_GDP_Rankings 2024_Top 10_Countries_Ranked_GDP_Forbes_India.pdf"
        ],
        "chunk_token_size": 1000,
        "model": config_list[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

solver = autogen.AssistantAgent(
    name="solver",
    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

coder = AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

reviewer = autogen.AssistantAgent(
    name="Code_Reviewer",
    is_termination_msg=termination_msg,
    system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
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
@solver.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount} {quote_currency}"


PROBLEM_general = "What is GDP of USA and Germany? Give output in there owm country currency. Also output who has greater GDP amomg them."
PROBLEM_refine = "What are the GDP figures for the USA and Germany, expressed in their respective national currencies? Additionally, determine which country has the higher GDP."


def _reset_agents():
    boss.reset()
    boss_aid.reset()
    coder.reset()
    reviewer.reset()


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss, boss_aid, coder, reviewer, solver], messages=[], max_round=12, speaker_selection_method="auto"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        problem=PROBLEM_general,
        n_results=3,
    )
    
rag_chat()