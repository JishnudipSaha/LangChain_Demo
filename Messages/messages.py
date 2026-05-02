from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

model = ChatOllama(model="gemma4:31b-cloud", temperature=0.7)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain"),
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print(messages)