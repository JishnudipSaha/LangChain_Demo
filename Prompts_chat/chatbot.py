from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()
model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)

# chat history / context history
chat_history = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    question = input("Ask a question: ")
    question = question.lower()
    # adding human messgage to the context
    chat_history.append(HumanMessage(content = question))
    
    # checking terminating command
    if question in ["exit", "quite"]:
        break
    
    # calling/invoking the model
    response = model.invoke(chat_history)
    
    # adding AI generated messgage to the context
    chat_history.append(AIMessage(content=response.content))
    
    # printing the AI response
    print("AI: ", response.content)
    print("-"*20)