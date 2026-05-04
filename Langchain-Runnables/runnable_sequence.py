from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


model = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0.7
)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Explain the following joke: {joke}',
    input_variables=['joke']
)
parser = StrOutputParser()

chain1 = RunnableSequence(prompt1, model, parser)
chain2 = RunnableSequence(prompt2, model, parser)
final_chain = RunnableSequence(chain1, chain2)

response = final_chain.invoke(
    {
        'topic': 'programming'
    }
)

print(response)

