import getpass
import os
from typing import List
from IPython.display import Markdown
from google import genai
from google.genai import types
from langchain_core.tools import tool
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
# Shopping list
shopping_list = []
# Functions
@tool
def add_shopping_items(items: List[str]):
    """Add multiple items to the shopping list."""
    for item in items:
        shopping_list.append(item)
    return {"status": "ok", "added": items}
@tool
def list_shopping_items():
    """Return all items currently in the shopping list."""
    return {"shopping_list": shopping_list}
# Configuration
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)
tools = [add_shopping_items, list_shopping_items]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps manage shopping lists. "
               "Use the available tools to add items to the shopping list "
               "or list the current items when requested by the user."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
# Create the Agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# User input
user_input = (
    "Hey there! I'm planning to bake a chocolate cake later today, "
    "but I realized I'm out of flour and chocolate chips. "
    "Could you please add those items to my shopping list?"
)
# Send the user input to Gemini
response = agent_executor.invoke({"input": user_input})
