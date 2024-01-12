import autogen
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import Annotated

config_list_gpt35 = autogen.config_list_from_json(
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


llm_config = {"config_list": config_list_gpt35, 
              "cache_seed": 42,
              "timeout": 120}




user_proxy = autogen.UserProxyAgent(
   name="User_proxy",
   system_message="A human admin.",
   code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
   human_input_mode="TERMINATE"
)
coder = autogen.AssistantAgent(
    name="Coder",
    llm_config=llm_config,
)
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)
groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# print('AssistantAgent_tools', coder.llm_config["tools"], pm.llm_config["tools"])
print("groupchat_tools",groupchat.llm_config["tools"])
print("manager_tools",manager.llm_config["tools"])

user_proxy.initiate_chat(manager, message="Find a latest paper about gpt-4 on arxiv and find its potential applications in software.")
# type exit to terminate the chat



# chatbot = autogen.AssistantAgent(
#     name="chatbot",
#     system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
#     llm_config=llm_config,
# )

# # create a UserProxyAgent instance named "user_proxy"
# user_proxy = autogen.UserProxyAgent(
#     name="user_proxy",
#     is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=10,
# )


# CurrencySymbol = Literal["USD", "EUR"]


# def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
#     if base_currency == quote_currency:
#         return 1.0
#     elif base_currency == "USD" and quote_currency == "EUR":
#         return 1 / 1.1
#     elif base_currency == "EUR" and quote_currency == "USD":
#         return 1.1
#     else:
#         raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")


# @user_proxy.register_for_execution()
# @chatbot.register_for_llm(description="Currency exchange calculator.")
# def currency_calculator(
#     base_amount: Annotated[float, "Amount of currency in base_currency"],
#     base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
#     quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
# ) -> str:
#     quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
#     return f"{quote_amount} {quote_currency}"


# # print("tools",chatbot.llm_config["tools"])

# # start the conversation
# user_proxy.initiate_chat(
#     chatbot,
#     message="How much is 700.45 USD in EUR?",
# )