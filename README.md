# LangChain Demo Workspace

This repository contains practical LangChain experiments and examples organized by topic.

## Project Structure

- `Basics/` - foundational examples
- `Langchains/` - simple, sequential, parallel, and conditional chain demos
- `Langchain-Runnables/` - runnable and notebook-based chain experiments
- `Messages/` - message handling examples
- `Output_parsers/` - output parsing examples
- `Prompts_chat/` - prompt and chat examples
- `Structured_Output/` - structured output workflows

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add required environment variables in a local `.env` file (not committed).

## Run

Run any script directly, for example:

```bash
python Langchains\simplechain.py
python Langchains\sequential_chain.py
python Langchains\parallel_chain.py
python Langchains\conditional_chain_updated.py
```

## Notes

- This repo is intended for learning and demos.
- Different scripts may require different model/provider setup (for example, Ollama or cloud API keys).

## Notebook Logic (`Langchain-Runnables/llm-amm-zindagi.ipynb`)

The notebook demonstrates a minimal chain flow:
1. `FakePromptTemplate.format(kwargs)` injects variables into a template.
2. `FakeLLM.predict(prompt)` returns a generated **string** response.
3. `FakeLLMChain.run(kwargs)` formats the prompt and returns the LLM response.

Important logic rule: keep return types consistent between `FakeLLM.predict` and `FakeLLMChain.run` (string-to-string, or dict-to-dict access if you intentionally wrap output).

## Runnable Notebook Logic (`Langchain-Runnables/langchain-mentos-zindagi.ipynb`)

This notebook demonstrates a Runnable-style pipeline:
1. A base `Runnable` interface defines `invoke(input_data)`.
2. `FakePromptTemplate`, `FakeLLM`, and `FakeStrOutputParser` each implement `invoke`.
3. `RunnableConnector([template, llm, parser])` passes output step-by-step through the chain.

This pattern keeps prompt building, model response, and output parsing modular and composable.
