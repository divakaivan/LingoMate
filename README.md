## Reproduction

- Clone the repo, and run `make` in the terminal. You should see something similar to:

```
Usage: make [option]

Options:
  help                 Show this help message
  setup                Install requirements, build docker services and prepare elasticsearch index
  db                   Setup database
  index                Prepare elasticsearch index
  start                Start docker services (detached mode) and run the app
  tracing              Start LangMate - RAG tracing
  stop                 Stop docker services
```

- Follow the setup steps (setup -> db -> start)

- Start ollama locally
    - Download [ollama](https://ollama.com/)
    - Run `ollama pull llama3.1` in your terminal. This will pull and setup llama3.1 for the RAG app 