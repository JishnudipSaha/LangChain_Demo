from PIL.ImImagePlugin import j
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

#--------------------------------------------------------------------------------

def word_count(input: str) -> int:
    return len(input.split())

word_count_chain = RunnableLambda(func=word_count)

model = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0.7
)

# joke geeneration chain
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# joke explanation chain
prompt2 = PromptTemplate(
    template='Explain the following joke: {joke}',
    input_variables=['joke']
)

# paser as String Parser
parser = StrOutputParser()

# joke generation chain
joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'count': word_count_chain
    }
)
full_chain = RunnableSequence(joke_gen_chain, parallel_chain)

response = full_chain.invoke(
    {
        'topic': 'Girl Friend'
    }
)

# print('Joke:', response['joke'])
# print('Count:', response['count'])

result = '''The Joke is: \n{joke} \nThe word count of the joke is: {count}'''.format(joke=response['joke'], count=response['count'])
print(result)

#--------------------------------------------------------------------------------