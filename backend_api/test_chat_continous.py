from openai import AsyncOpenAI
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# client = AsyncOpenAI()
# client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client = AsyncOpenAI(api_key=openai_api_key)
# Construct the system prompt
system_prompt_template = \
"""You are Bobby, a virtual assistant create by Guosong Li. Today is {today}. You provide responses to questions that are clear, straightforward, and factually accurate, without speculation or falsehood. Given the following context, please answer each question truthfully to the best of your abilities based on the provided information. Answer each question with a brief summary followed by several bullet points. 

Example:
Summary of answer
- bullet point 1
- bullet point 2
...

<context>
{context}
</context>
"""
# print("debug 001")
# with open("news_result.txt") as in_file:
#     context_content = in_file.read()

with open("2024-05-EB-A_Compact_GuideTo_RAG.md", "r", encoding="utf-8", errors="ignore") as in_file:
    context_content = in_file.read()
# print("debug 002")

# with open(["2024-05-EB-A_Compact_GuideTo_RAG.md","2023-10-EB-Big-Book-of-MLOps-2nd-Edition.md"], "r", encoding="utf-8", errors="ignore") as in_file:
#     context_content = in_file.read()

system_prompt = system_prompt_template.format(
    context=context_content, 
    today=datetime.today().strftime('%Y-%m-%d')
)

# print("debug 003")

async def chat_func(history):

    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}] + history,
        max_tokens=256,
        temperature=0.5,
        stream=True,
    )
    print("debug 011")

    buffer = ""
    async for r in result:
        # print("debug 012")
        next_token = r.choices[0].delta.content
        # print("debug 013")
        if next_token:
            # print("debug 014")
            print(next_token, flush=True, end="")
            # print("debug 015")
            buffer += next_token
            # print("debug 016")

    print("\n", flush=True)
    # print("debug 017")

    return buffer

# print("debug 004")

async def continous_chat():
    history = []

    # Loop to receive user input continously
    while(True):
        user_input = input("> ")
        if user_input == "exit":
            break
        # print("debug 006")

        history.append({"role": "user", "content": user_input})
        # print("debug 007")

        # notice every time we call the chat function
        # we pass all the history to the API
        bot_response = await chat_func(history)
        # print("debug 008")

        history.append({"role": "assistant", "content": bot_response})
print("debug 005")

asyncio.run(continous_chat())



