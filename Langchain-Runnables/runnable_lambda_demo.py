from langchain_core.runnables import RunnableLambda


def word_count(input: str) -> int:
    return len(input.split())

word_count_chain = RunnableLambda(func=word_count)
response = word_count_chain.invoke('Hello Jishnudip!')
print(response)