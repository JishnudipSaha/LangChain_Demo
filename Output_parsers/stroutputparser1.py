from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)

# 1st prompt: Detailed report
template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"]
)

# 2nd prompt: Summary of the report
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Black Hole"})
print(result)