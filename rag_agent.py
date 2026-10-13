import os
import csv
import sqlite3
import argparse
import warnings
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Native document parsers
import pypdf
import docx2txt

# LangChain / LangGraph components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document

# New LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import ollama

# ==========================================
# Model & Document Loading
# ==========================================
def ensure_model_exists(model_name):
    print(f"System: Checking availability for '{model_name}'...")
    try:
        local_models_response = ollama.list()
        
        if isinstance(local_models_response, dict) and 'models' in local_models_response:
            local_models = [m['name'] for m in local_models_response['models']]
        else:
            local_models = [m.model for m in local_models_response.models]
            
        if model_name not in local_models and f"{model_name}:latest" not in local_models:
            print(f"System: Downloading '{model_name}' (this may take a few minutes)...")
            ollama.pull(model_name)
            print(f"System: Successfully downloaded '{model_name}'.")
        else:
            print(f"System: '{model_name}' is ready.")
            
    except Exception as e:
        print(f"\n[Error] Failed to communicate with Ollama: {e}")
        exit(1)

def process_image_with_qwen(file_path):
    print(f"  [Vision] Analyzing image: {file_path}")
    vision_model = 'qwen2.5vl:7b'
    ensure_model_exists(vision_model)
    try:
        response = ollama.chat(
            model=vision_model,
            messages=[{'role': 'user', 'content': 'Extract all text from this image exactly as written. If there are charts or diagrams, describe them.', 'images': [file_path]}]
        )
        return [Document(page_content=response['message']['content'], metadata={"source": file_path})]
    except Exception as e:
        print(f"  [Error] Failed to process image {file_path}: {e}")
        return []

def load_pdf(file_path):
    docs = []
    reader = pypdf.PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text, metadata={"source": file_path, "page": i}))
    return docs

def load_docx(file_path):
    return [Document(page_content=docx2txt.process(file_path), metadata={"source": file_path})]

def load_csv(file_path):
    docs = []
    with open(file_path, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            docs.append(Document(page_content="\n".join(f"{k}: {v}" for k, v in row.items()), metadata={"source": file_path, "row": i}))
    return docs

def load_documents(path):
    docs = []
    files = [path] if os.path.isfile(path) else [os.path.join(path, f) for f in os.listdir(path)]
    image_extensions = {'.png', '.jpg', '.jpeg'}

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                print(f"  Loading PDF: {file_path}...")
                docs.extend(load_pdf(file_path))
            elif ext in ['.docx', '.doc']:
                print(f"  Loading Word Doc: {file_path}...")
                docs.extend(load_docx(file_path))
            elif ext == '.csv':
                print(f"  Loading CSV: {file_path}...")
                docs.extend(load_csv(file_path))
            elif ext in image_extensions:
                docs.extend(process_image_with_qwen(file_path))
        except Exception as e:
            print(f"  [Error] Could not read {file_path}: {e}")
    return docs

# ==========================================
# Define the LangGraph State & Nodes
# ==========================================
# The 'State' tracks the conversation and the retrieved context across nodes
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


def create_graph_workflow(llm, retriever):
    """Factory function to build the graph using specific model and retriever instances."""

    def retrieve_node(state: State):
        messages = state["messages"]
        question = messages[-1].content

        if len(messages) > 1:
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Given the chat history and the latest user question, formulate a standalone query to search a database. Do NOT answer it. Just output the search string."),
                MessagesPlaceholder("messages")
            ])
            search_query = (rewrite_prompt | llm).invoke({"messages": messages}).content
        else:
            search_query = question

        docs = retriever.invoke(search_query)
        return {"context": "\n\n".join([doc.page_content for doc in docs])}

    def generate_node(state: State):
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assistant for question-answering tasks. Use the following retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n\nContext:\n{context}"),
            MessagesPlaceholder("messages")
        ])

        chain = qa_prompt | llm

        response_content = ""
        for chunk in chain.stream({"messages": state["messages"], "context": state["context"]}):
            response_content += chunk.content
            print(chunk.content, end="", flush=True)

        return {"messages": [{"role": "assistant", "content": response_content}]}

    # Compile workflow inside the factory
    workflow = StateGraph(State)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)


# ==========================================
# Main App Setup
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Local LangGraph RAG Agent with Qwen")
    parser.add_argument("--path", type=str, default=None, help="Path to documents.")
    parser.add_argument("--list-sessions", action="store_true", help="List all previous session IDs and exit.")
    parser.add_argument("--model", type=str, default="qwen3:8b", help="Reasoning agent model.")
    parser.add_argument("--session", type=str, default=None, help="Load a previous session ID. Use --list-sessions "
                                                                  "to list all previous session IDs.")
    args = parser.parse_args()

    if args.list_sessions:
        if not os.path.exists("chat_history.db"):
            print("No chat history database found. Start a session first!")
            return

        try:
            conn = sqlite3.connect("chat_history.db")
            cursor = conn.cursor()
            # LangGraph stores the session ID in the thread_id column of the checkpoints table
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC")
            rows = cursor.fetchall()

            if not rows:
                print("No previous sessions found in the database.")
            else:
                print("\nAvailable Previous Sessions:")
                print("-" * 30)
                for row in rows:
                    print(f" • {row[0]}")
                print("-" * 30)
            conn.close()
        except sqlite3.Error as e:
            print(f"Error reading the database: {e}")
        return

    if not args.path:
        print("Error: --path is required to start the agent. (e.g., --path my_docs/)")
        return

    ensure_model_exists(args.model)
    ensure_model_exists("nomic-embed-text")

    print("\nProcessing documents...")
    docs = load_documents(args.path)
    if not docs:
        print("\nNo valid documents were successfully loaded. Exiting.")
        return

    llm = ChatOllama(model=args.model, temperature=0)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    print("\nBuilding vector database...")
    vectorstore = Chroma.from_documents(documents=text_splitter.split_documents(docs), embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ==========================================
    # Compile the Graph
    # ==========================================

    app = create_graph_workflow(llm, retriever)

    # ==========================================
    # Interactive Terminal Loop
    # ==========================================
    print("\n" + "="*50)
    print(f"LangGraph Agent initialized with model: {args.model}")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("="*50 + "\n")

    session_id = args.session if args.session else datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{'Resuming' if args.session else 'Started new'} session: {session_id}\n")
    
    # The config dictionary tells the graph which SQLite record to read/write to
    config = {"configurable": {"thread_id": session_id}}

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Ending conversation. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        print("\nAgent: ", end="", flush=True)
        # Execute the graph. The streaming text is handled inside the generate_node
        app.invoke({"messages": [("human", user_input)]}, config=config)
        print("\n")

if __name__ == "__main__":
    main()