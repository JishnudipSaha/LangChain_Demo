from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Initialize the Chat Model
chat_model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7,
    
)

# 2. Define your conversation context
messages = [
    SystemMessage(content="You are a helpful Data Science assistant."),
    HumanMessage(content="Explain the difference between a CNN and a Transformer.")
]

# 3. Invoke the model
response = chat_model.invoke("tell 5 line about cricket.")
print(f"Chat Output: {response.content}")