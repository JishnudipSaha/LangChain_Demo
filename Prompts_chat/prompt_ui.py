from itertools import chain

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from regex import template
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
load_dotenv()


chat_model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7,
)

st.header('Research tool')
# user_input = st.text_input('Enter your prompt')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# response = chat_model.invoke(user_input)

# filling the placeholders

template = load_prompt('template.json')

# prompt = template.invoke({
#     "paper_input": paper_input,
#     "style_input": style_input,
#     "length_input": length_input
# })


if st.button('Summarize'):
    chain = template | chat_model
    response = chain.invoke(
        {
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
        }
    )
    # response = chat_model.invoke(prompt)
    st.write(response.content)
