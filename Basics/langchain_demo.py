from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="ollama:gemma4:31b-cloud",
    tools=[],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "can you tell me the capital of India?"}]}
)
print(result["messages"][-1].content_blocks)