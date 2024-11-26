# LLM RAG
This project is a proof of concept for knowledge training and retrieval using local or remote systems.
Current implementation is using Jupiter Notebooks with a goal of making API baised.

## Technology
### Application Utilized 
 - Ollama - Interface with multiple LLM and embedding models
 - Postgres - Persistence 
 - PGVector - Extension to add Vector queries to Postgres

## Getting started
### Install Dependencies
```python
pip install -r requirements.txt
```

## TODOs
 - [ ] Make API
 - [ ] Containerize Application
 - [ ] Converstion Chain (ConverstationMemoryBuffer)


## Helpful Links
[Run Huggingfaces GGUF models in Ollama](https://huggingface.co/docs/hub/en/ollama)

[Reddis as a VectorDB](https://cookbook.openai.com/examples/vector_databases/redis/getting-started-with-redis-and-openai)

[IBM Granite](https://ollama.com/blog/ibm-granite)