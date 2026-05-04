# LangChain Demo Workspace

This repository is a comprehensive educational workspace demonstrating LangChain concepts, moving from basic LLM primitives to complex LCEL (LangChain Expression Language) orchestration.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add required environment variables in a local `.env` file (not committed).

## Project Structure & Learning Path

The project is organized as a progressive learning roadmap:

### 🟢 Level 1: Fundamentals
- **`Basics/`**: Getting started with LLM initialization.
    - **Goal**: Learn how to initialize and swap between providers (Ollama, Google, HuggingFace).
    - **Key Files**: `langchain_demo.py`, `ollama_chat.py`.
- **`Messages/`**: Understanding the conversation loop.
    - **Goal**: Master the roles of `SystemMessage`, `HumanMessage`, and `AIMessage` to structure conversations.
    - **Key Files**: `messages.py`.

### 🟡 Level 2: Data Handling & Control
- **`Prompts_chat/`**: Designing interactive interfaces.
    - **Goal**: Implement chat context and dynamic message placeholders for stateful interactions.
    - **Key Files**: `chatbot_context.py`, `message_placeholder.py`.
- **`Output_parsers/`**: Transforming raw text into usable data.
    - **Goal**: Convert LLM responses into structured strings or Pydantic objects.
    - **Key Files**: `pydanticoutputparser.py`, `stroutputparser.py`.
- **`Structured_Output/`**: Leveraging native LLM capabilities.
    - **Goal**: Use `.with_structured_output()` for high-reliability schema adherence using Pydantic and TypedDict.
    - **Key Files**: `with_structured_output_pydantic.py`.

### 🔴 Level 3: Advanced Orchestration
- **`Langchain-Runnables/`**: The building blocks of LCEL.
    - **Goal**: Understand the Runnable interface and how components are piped together.
    - **Key Files**: `runnable_sequence.py`, `runnable_parallel.py`, `runnable_passthrough.py`, `runnable_branch.py`, `runnable_lambda.py`.
- **`Langchains/`**: Complex logic and workflows.
    - **Goal**: Build sequential and conditional chains using LCEL (Sequential, Parallel, and Conditional routing).
    - **Key Files**: `sequential_chain.py`, `conditional_chain.py`.

## Execution Guide

### 🚀 Quick Start (Fundamentals)
```bash
python Basics/langchain_demo.py
python Basics/ollama_chat.py
python Messages/messages.py
```

### 🛠️ Data & Structure Testing
```bash
# Testing structured outputs and parsers
python Output_parsers/pydanticoutputparser.py
python Structured_Output/with_structured_output_pydantic.py
```

### 🧠 Conversation & State
```bash
# Testing chat memory and context
python Prompts_chat/chatbot_context.py
python Prompts_chat/chatbot.py
```

### ⛓️ Advanced Chain Logic
```bash
# Testing LCEL Sequential and Conditional flows
python Langchains/sequential_chain.py
python Langchains/conditional_chain.py
```

## Notebook Logic

### `Langchain-Runnables/llm-amm-zindagi.ipynb`
The notebook demonstrates a minimal chain flow:
1. `FakePromptTemplate.format(kwargs)` injects variables into a template.
2. `FakeLLM.predict(prompt)` returns a generated **string** response.
3. `FakeLLMChain.run(kwargs)` formats the prompt and returns the LLM response.

Important logic rule: keep return types consistent between `FakeLLM.predict` and `FakeLLMChain.run` (string-to-string, or dict-to-dict access if you intentionally wrap output).

### `Langchain-Runnables/langchain-mentos-zindagi.ipynb`
This notebook demonstrates a Runnable-style pipeline:
1. A base `Runnable` interface defines `invoke(input_data)`.
2. `FakePromptTemplate`, `FakeLLM`, and `FakeStrOutputParser` each implement `invoke`.
3. `RunnableConnector([template, llm, parser])` passes output step-by-step through the chain.

This pattern keeps prompt building, model response, and output parsing modular and composable.

## Notes
- This repo is intended for learning and demos.
- Different scripts may require different model/provider setup (for example, Ollama or cloud API keys).
