from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

model_name = os.getenv("MODEL_NAME")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def hello_llm(user_input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are concise and helpful assistant."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.2
    )
    return response.choices[0].message.content


def main():
    print(hello_llm("What is RAG?"))
    print(hello_llm("What is the use of Temperature in LLM?"))
    print(hello_llm("What is the use of Top P in LLM?"))

if __name__ == "__main__":
    main()
