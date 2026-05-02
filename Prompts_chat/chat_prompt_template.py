from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms what is {topic}")
    # SystemMessage(content="You are a helpful {domain} expert."),
    # HumanMessage(content="Explain in simple terms what is {topic}")
])

prompt = chat_template.invoke({
    "domain": "Cricket Expert",
    "topic": "Stump-Out"
})

print(prompt)