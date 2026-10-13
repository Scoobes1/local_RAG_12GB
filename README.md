# Local RAG Agent with Qwen Models & LangGraph

A robust, fully local Retrieval-Augmented Generation (RAG) conversational agent written in Python. It uses LangGraph for stateful orchestration, Ollama for running local Qwen models, ChromaDB for vector storage, and SQLite for persistent chat memory—all optimized to run entirely offline within the constraints of a 12GB VRAM GPU (such as the RTX 4070).
Key Features

- Stateful LangGraph Workflow: Built on modern graph-based state machines, featuring automatic query rewriting for context-aware multi-turn conversations.


- Persistent SQLite Memory: Chat histories are automatically saved to a local database (chat_history.db), allowing you to resume previous conversations anytime.


- Multi-Modal Document Support: Native parsing for PDFs, Word documents (.docx), and CSVs, plus visual extraction for Images (.png, .jpg) using Qwen-VL (**not tested yet**).


- Automatic Model Management: Automatically checks if your chosen Qwen model or embedding model is available locally, and pulls them from Ollama if missing.


## Hardware Optimization (12GB VRAM Limit)

The default model configuration (qwen3:8b) uses roughly 5.5GB of VRAM in 4-bit quantization, leaving over 6GB of headroom for document chunking, embeddings (nomic-embed-text), and a large context window.
Prerequisites & Installation

# Installation

## 1. Install Python and Ollama

Ensure Python (tested on v3.13) and Ollama is installed and running in the background on your machine.

## 2. Install Python Dependencies

It's recommended you create a virtual environment before installation of the relevant libraries. 

Open your terminal inside the project directory and install the environment:

    pip install -r requirements.txt

# Running the LLM

## Basic Run (Start a New Session)

Provide a path to a single document or an entire folder:

    python rag_agent.py --path ./my_documents/

An example pdf is included in the docs folder. To chat with this document:

    python rag_agent.py --path docs/LLMs_forProteinDesign.pdf

## Choose a Specific Model / Quantization

You can select alternative models that safely fit within your 12GB VRAM limit using the --model flag:

    python rag_agent.py --path ./my_documents/ --model qwen2.5:7b-instruct-q8_0

Available options include: qwen3:8b (default), qwen3:8b-instruct-q8_0, qwen3.5:9b, 
qwen2.5:14b-instruct-q4_K_M, and qwen2.5vl:7b.

## Sessions

List previous Sessions:

    python rag_agent.py --list-sessions

Pick a session ID example:

    python rag_agent.py --path ./my_documents/ --session 20261001_165230

## Exiting the Chat

While inside an active conversation session, type quit, exit, or q to close the program safely.
