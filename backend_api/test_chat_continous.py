from openai import AsyncOpenAI
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

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

file_list = [
    "2024-05-EB-A_Compact_GuideTo_RAG.md",
    "What is Retrieval-Augmented Generation (RAG).txt",
    "What Is Retrieval.txt",
]

# To store all file contents in a list
context_content = " " 

for file_name in file_list:
    try:
        with open(file_name, "r", encoding="utf-8", errors="ignore") as in_file:
            content = in_file.read()
            context_content = context_content + content  # Store the content
            print(f"Read content from {file_name}")

    except FileNotFoundError:
        print(f"File {file_name} not found!")
    except Exception as e:
        print(f"An error occurred with {file_name}: {e}")

system_prompt = system_prompt_template.format(
    context=context_content, 
    today=datetime.today().strftime('%Y-%m-%d')
)

async def chat_func(history):

    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}] + history,
        max_tokens=256,
        temperature=0.5,
        stream=True,
    )

    buffer = ""
    async for r in result:
        next_token = r.choices[0].delta.content
        if next_token:
            print(next_token, flush=True, end="")
            buffer += next_token

    print("\n", flush=True)

    return buffer

async def continous_chat():
    history = []

    # Loop to receive user input continously
    while(True):
        user_input = input("> ")
        if user_input == "exit":
            break

        history.append({"role": "user", "content": user_input})

        # notice every time we call the chat function
        # we pass all the history to the API
        bot_response = await chat_func(history)

        history.append({"role": "assistant", "content": bot_response})

asyncio.run(continous_chat())
