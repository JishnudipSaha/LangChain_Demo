from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()


model = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0.7
)

prompt1 = PromptTemplate(
    template='Generate a Twitter post about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a LinkedIn post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model, parser),
        'linkedin': RunnableSequence(prompt2, model, parser)
    }
)


result = parallel_chain.invoke(
    {
        'topic': 'AI'
    }
)
print('Tweet:', result['tweet'])
print('LinkedIn:', result['linkedin'])