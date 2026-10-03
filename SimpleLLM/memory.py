import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

# System prompt
system_prompt = """
You are the assistant for a tiny candle shop. 

Step 1:Check whether the user mentions either of our candles:
   • Forest Breeze (woodsy scent, 40 h burn, $18)  
   • Vanilla Glow (warm vanilla, 35 h burn, $16)

Step 2:List any assumptions the user makes
   (e.g. "Vanilla Glow lasts 50 h" or "Forest Breeze is unscented").

Step 3:If an assumption is wrong, correct it politely.  
   Then answer the question in a friendly tone.  
   Mention only the two candles above-we don't sell anything else.

Use exactly this output format:
Step 1:<your reasoning>
Step 2:<your reasoning>
Step 3:<your reasoning>
Response to user: <final answer>
"""

# start chat_history
chat_history = []

# first message
user_input = "I would like to buy 1 Forest Breeze. Can I pay $10?"
full_content = f"System instructions: {system_prompt}\n\n chat_history: {chat_history}\n\n user_message: {user_input}"
response = client.models.generate_content(
    model = "gemini-3.6-flash",
    contents = full_content
)

# Append to chat history
chat_history.append({"role": "user", "content": user_input})
chat_history.append({"role": "assisstant", "content": response.text})

# second message
user_input = "What did I say I wanted to buy?"
full_content = f"System instructions: {system_prompt}\n\n chat_history: {chat_history}\n\n user_message: {user_input}"
response = client.models.generate_content(
    model = "gemini-3.6-flash",
    contents = full_content
)

# Append to chat history
chat_history.append({"role": "user", "content": user_input})
chat_history.append({"role": "assisstant", "content": response.text})

print(response.text)
