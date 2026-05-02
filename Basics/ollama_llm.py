from langchain_ollama import OllamaLLM

# 1. Initialize the LLM
llm = OllamaLLM(model="gemma4:31b-cloud")

# 2. Direct completion
response = llm.invoke("The capital of West Bengal is")
print(f"LLM Output: {response}")