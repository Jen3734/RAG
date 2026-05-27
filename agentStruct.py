import os

from agentCompress import build_llm
from langchain_core.prompts import ChatPromptTemplate

def input_guardrail():
    pass

def context_guardrail():
    pass

def response_guardrail():
    pass

def get_user_input():
    pass

def get_context():
    pass

def get_response(user_input, context):
    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "What is the capital of France?"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"user_input": user_input, "context": context})
    return response

def save_context():
    pass

def save_response():
    pass

def main():
    input_guardrail()
    user_input = get_user_input()
    context = get_context()
    response = get_response(user_input, context)
    save_context()
    save_response()

if __name__ == "__main__":
    main()

