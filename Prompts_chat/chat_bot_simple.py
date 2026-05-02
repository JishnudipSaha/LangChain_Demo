from itertools import chain

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from regex import template
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
load_dotenv()


chat_model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7,
)

st.header('Refund Assistant')

chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful customer support assistant'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

user_input = st.text_input('Enter your prompt')
chat_history = []
# chat_history.extend(HumanMessage(content=user_input))

if st.button('Ask'):
    # loading the chat history
    with open('./Prompts_chat/simple_chat_history.txt', 'r') as f:
        chat_history.extend(f.readlines())
        
    # creating chain to pass the prompt to the model
    chain = chat_template | chat_model
    response = chain.invoke(
        {
        "chat_history": chat_history,
        "query": user_input
        }
    )
    # response = chat_model.invoke(prompt)
    # chat_history.extend(AIMessage(content=response.content))
    with open('./Prompts_chat/chat_history.txt', 'a') as f:
        f.write(f"HumanMessage(content='{user_input}')\n")
        f.write(f"AIMessage(content='{response.content}')\n")
        # f.write(response.content)
    st.write(response.content)