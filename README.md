# LangChain Demo Workspace

This repository contains practical LangChain experiments and examples organized by topic.

## Project Structure

- `Basics/` - foundational examples
- `Langchains/` - simple, sequential, parallel, and conditional chain demos
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
