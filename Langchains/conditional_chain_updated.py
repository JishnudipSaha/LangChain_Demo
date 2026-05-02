from gitdb import typ
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()


model = ChatOllama(
    model = "gemma4:31b-cloud",
    temperature=0.7
)

parser = StrOutputParser()

# schema: pydantic class
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

pydanticParser = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={
        'format_instructions': pydanticParser.get_format_instructions()
    }
)

sentiment_classification_chain = RunnablePassthrough.assign(
    sentiment=(prompt1 | model | pydanticParser | RunnableLambda(lambda x: x.sentiment))
)

prompt2 = PromptTemplate(
    template='Write a appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write a appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

result = sentiment_classification_chain.invoke(
    {
        'feedback': 'I am facing a huge battery issue and also heating issue, i this phone 2 months ago.'
    }
)
print("The sentiment is: ", result['sentiment'])
print(type(result))

branch_chain = RunnableBranch(
    # (condition, chain)
    (
        lambda x: x['sentiment'] == 'positive', # condition
        RunnableLambda(lambda x: {'feedback': x['feedback']}) | prompt2 | model | parser # chain to be executed
    ),
    (
        lambda x: x['sentiment'] == 'negative',
        RunnableLambda(lambda x: {'feedback': x['feedback']}) | prompt3 | model | parser
    ),
    # default chain
    RunnableLambda(lambda x: 'Could not find sentiment')
)


final_chain = sentiment_classification_chain | branch_chain

response = final_chain.invoke(
    {
        'feedback': 'I am facing a huge battery issue and also heating issue, i this phone 2 months ago.'
    }
)

print(response)
final_chain.get_graph().print_ascii()
