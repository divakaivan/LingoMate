## LingoMate Introduction 

https://www.notion.so/Summary-3f194f4e03e5485fbbcdc5fd11892c57

## User Flow 
![Sample Image](/UserFlow.png)


## Architecture
![Sample Image](/Diagram.png)

## Monitoring 

![Sample Image](/Monitoring.png)

## RAG Tracing  

![Sample Image](/RAG_Tracing.png)


## Reproduction

- Clone the repo, and run `make` in the terminal. You should see something similar to:

```
Usage: make [option]

Options:
  help                 Show this help message
  setup                Install requirements and build docker services
  db                   Setup database
  start                Start docker services (detached mode)
  stop                 Stop docker services
```

- Follow the setup steps (setup -> db -> start)

- Start ollama locally
    - Download [ollama](https://ollama.com/)
    - Run `ollama pull llama3.1` in your terminal. This will pull and setup llama3.1 for the RAG app 

