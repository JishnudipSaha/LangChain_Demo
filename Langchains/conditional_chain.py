from gitdb import typ
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()


model = ChatOllama(
    model = "gemma4:31b-cloud",
    temperature=0.7
)

parser = StrOutputParser()

# pydantic class
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

classifier_chain = prompt1 | model | pydanticParser

prompt2 = PromptTemplate(
    template='Write a appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write a appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

result = classifier_chain.invoke(
    {
        'feedback': 'I am facing a huge battery issue and also heating issue, i this phone 2 months ago.'
    }
)
print("The sentiment is: ", result.sentiment)
print(type(result))

branch_chain = RunnableBranch(
    # (condition, chain),
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    # default chain
    RunnableLambda(lambda x: 'Couldnot find sentiment')
)


final_chain = classifier_chain | branch_chain

response = final_chain.invoke(
    {
        'feedback': 'I am facing a huge battery issue and also heating issue, i this phone 2 months ago.'
    }
)

print(response)
final_chain.get_graph().print_ascii()