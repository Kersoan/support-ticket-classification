import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def call_gpt_oss_20b(prompt: str) -> str:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-20b",
        temperature=0.7,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
    # return response

print("----------------------------------------")
# Example usage
print(call_gpt_oss_20b("Explain chennai in one sentence"))

