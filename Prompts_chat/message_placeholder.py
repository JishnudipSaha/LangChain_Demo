from langchain_core import chat_history
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
load_dotenv()

model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful customer support assistant'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

chat_history = []
#load current history
with open('./Prompts_chat/chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

# print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund id=#12345?'})
# print(prompt)

response = model.invoke(prompt)
print(response.content)