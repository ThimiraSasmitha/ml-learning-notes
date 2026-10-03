import os
from dotenv import load_dotenv
from typing import List
from google import genai
from google.genai import types
# Shopping list
shopping_list: List[str] = []
# Functions
def add_shopping_items(items: List[str]):
    """Add multiple items to the shopping list."""
    for item in items:
        shopping_list.append(item)
    return {"status": "ok", "added": items}
def list_shopping_items():
    """Return all items currently in the shopping list."""
    return {"shopping_list": shopping_list}
# Function declarations
add_shopping_items_declaration = {
    "name": "add_shopping_items",
    "description": "Add one or more items to the shopping list",
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of shopping items to add"
            }
        },
        "required": ["items"]
    }
}
list_shopping_items_declaration = {
    "name": "list_shopping_items",
    "description": "List all current items in the shopping list",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
# Configuration Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
tools = types.Tool(function_declarations=[
    add_shopping_items_declaration,
    list_shopping_items_declaration
])
config = types.GenerateContentConfig(tools=[tools])
# User input
user_input = (
    "Hey there! I'm planning to bake a chocolate cake later today, "
    "but I realized I'm out of flour and chocolate chips. "
    "Could you please add those items to my shopping list?"
)
# Send the user input to Gemini
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=user_input,
    config=config,
)
print("Model Output Function Call")
print(response.candidates[0].content.parts[0].function_call)
print("\n")
#Execute Function
tool_call = response.candidates[0].content.parts[0].function_call
if tool_call.name == "add_shopping_items":
    result = add_shopping_items(**tool_call.args)
    print(f"Function execution result: {result}")
elif tool_call.name == "list_shopping_items":
    result = list_shopping_items()
    print(f"Function execution result: {result}")
else:
    print(response.candidates[0].content.parts[0].text)
