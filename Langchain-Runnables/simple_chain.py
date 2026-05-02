from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from streamlit import user
load_dotenv()

model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)
user_input = "What is the capital of India?"

response = model.invoke(user_input)
print(response.content)