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


# PROBLEM_general = "What is GDP of USA and Germany? Give output in there owm country currency. Also output who has greater GDP amomg them."
PROBLEM = "What are the GDP figures for the USA and Germany? Additionally, determine which country has the higher GDP and output GDP in their respective national currencies"
    
builder = AgentBuilder(
    config_file_or_env=config_file_or_env, builder_model="gpt-3.5-turbo", agent_model="gpt-3.5-turbo"
)
building_task = "Generate some agents that can find read documents related to finance and solve task related to finance/economic domain. For example reading financial documents and comapring GDP for countries."
agent_list, agent_configs = builder.build(building_task, llm_config, coding=True, use_oai_assistant=True)

# print("*******agent_configs*******")
# # print(agent_configs['agent_configs'][0]['system_message'])
# saved_path = builder.save()
# print("agent_list", agent_list)
# print("*******agent_configs_done*******")
# for agent in agent_list:
#     print("agent__: ", agent)

# boss = autogen.UserProxyAgent(
#     name="Boss",
#     is_termination_msg=termination_msg,
#     human_input_mode="NEVER",
#     system_message="The boss who ask questions, give tasks and execute code accordingly.",
#     # code_execution_config=None,  # we do want to execute code in this case.
#     max_consecutive_auto_reply=10,
#     default_auto_reply="Reply `TERMINATE` if the task is done.",
# )
boss = agent_list[0]

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
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
    code_execution_config=False,  # we don't want to execute code in this case.
)

solver = autogen.AssistantAgent(
    name="solver",
    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

currency_aid = autogen.AssistantAgent(
    name="currency_aid",
    system_message="Suggest currency of given countries and convert it. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

# agent_list.insert(0, boss)
agent_list.extend([boss_aid, solver, currency_aid]) 
    
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


# @boss.register_for_execution()
# @solver.register_for_llm(description="Currency exchange calculator.")
# def currency_calculator(
#     base_amount: Annotated[float, "Amount of currency in base_currency"],
#     base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
#     quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
# ) -> str:
#     quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
#     return f"{quote_amount} {quote_currency}"


# print("********printing agent names********")
# print("solver_tools_function: ",solver.llm_config['tools'][0]['function'])
# print("********printing agent names done********")


def start_task(execution_task: str, agent_list: list):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12, speaker_selection_method="auto")
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **llm_config})
    # agent_list[0].initiate_chat(manager, message=execution_task)   
    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        problem=execution_task,
        n_results=1,
    )
    
# start_task(
#     execution_task=PROBLEM,
#     agent_list=agent_list,
# )

def _reset_agents():
    boss.reset()
    boss_aid.reset()
    solver.reset()


def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss, boss_aid, solver, currency_aid], messages=[], max_round=12, speaker_selection_method="auto"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        problem=PROBLEM,
        n_results=1,
    )

def call_rag_chat():
    _reset_agents()

    # In this case, we will have multiple user proxy agents and we don't initiate the chat
    # with RAG user proxy agent.
    # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    # it from other agents.
    def retrieve_content(message, n_results=3):
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.
        # Check if we need to update the context.
        update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and boss_aid.update_context:
            boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
            _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
        else:
            ret_msg = boss_aid.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message
    
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
    
    def currency_calculator(
        base_amount: Annotated[float, "Amount of currency in base_currency"],
        base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
        quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
    ) -> str:
        quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
        return f"{quote_amount} {quote_currency}"
    
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
            {'description': 'Currency exchange calculator.', 'name': 'currency_calculator', 
                'parameters': {'type': 'object', 
                'properties': {
                'base_amount': {'type': 'number', 'description': 'Amount of currency in base_currency'}, 
                'base_currency': {'enum': ['USD', 'EUR'], 'type': 'string', 'default': 'USD', 'description': 'Base currency'}, 
                'quote_currency': {'enum': ['USD', 'EUR'], 'type': 'string', 'default': 'EUR', 'description': 'Quote currency'}}, 
                'required': ['base_amount']
                }
            }
        ],
        "config_list": config_list,
        "timeout": 60,
        "cache_seed": 42,
    }
    
    

    for agent in [solver, currency_aid]:
        # update llm_config for assistant agents.
        agent.llm_config.update(llm_config)

    for agent in [boss, solver, currency_aid]:
        # register functions for all agents.
        agent.register_function(
            function_map={
                "retrieve_content": retrieve_content,
            }
        )
        agent.register_function(
            function_map={
                "currency_calculator": currency_calculator,
            }
        )
    for agent in [boss, solver, currency_aid]:
        print(f"agent_tools_function: ,{agent.llm_config}")

    groupchat = autogen.GroupChat(
        agents=[boss, solver, currency_aid],
        messages=[],
        max_round=12,
        speaker_selection_method="random",
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

print("rag_chat")    
rag_chat()
print("call_rag_chat")
call_rag_chat()