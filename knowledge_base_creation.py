import os
from typing import Dict, Optional, List
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings # Changed to OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Base directory for the raw knowledge base files
BASE_KB_DIR = "knowledge_base"

# Base directory for the ChromaDB persistent storage
BASE_CHROMA_DB_DIR = "chroma_db"

# Ollama model for embeddings.
# IMPORTANT: You MUST have pulled this model using 'ollama pull nomic-embed-text'
# from your terminal before running this script.
EMBEDDING_MODEL = "nomic-embed-text"

# Ollama LLM model name (for potential future agent use, included here for completeness)
# IMPORTANT: You MUST have pulled this model (e.g., 'ollama pull llama3')
LLM_MODEL = "mistral" # Or 'mistral', 'phi3', etc.

# Function to load documents from a directory supporting multiple types
def load_documents_from_dir(folder_path: str) -> List[any]: # Returns List[Document] but type hint 'Document' not universally available
    """
    Loads documents from a specified directory, supporting .txt, .md, .pdf, .ppt, .pptx files.

    Args:
        folder_path (str): The path to the directory containing the documents.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    documents = []
    print(f"Loading documents from: {os.path.abspath(folder_path)}")

    if not os.path.exists(folder_path):
        print(f"Error: Directory for knowledge base not found: {folder_path}. Please create it and place your documents inside.")
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            loader = None
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
            elif filename.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            elif filename.endswith(".pdf"):
                loader = UnstructuredPDFLoader(file_path)
            elif filename.endswith((".ppt", ".pptx")):
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue # Skip to the next file

            if loader:
                try:
                    print(f"Loading {filename}...")
                    loaded_docs = loader.load()
                    if loaded_docs:
                        documents.extend(loaded_docs)
                    else:
                        print(f"No content loaded from {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return documents

# Function to create a vector database for a specific phase
def create_vector_db(phase: str) -> Optional[str]:
    """
    Creates a Chroma vector database for a given SDLC phase using OllamaEmbeddings.

    Args:
        phase (str): The name of the SDLC phase (e.g., "requirements_phase").

    Returns:
        Optional[str]: The path to the created persistent ChromaDB directory, or None if creation failed.
    """
    phase_kb_path = os.path.join(BASE_KB_DIR, phase)
    persist_directory = os.path.join(BASE_CHROMA_DB_DIR, f"{phase}_security_kb") # Consistent naming

    print(f"\n--- Processing Phase: {phase} ---")
    print(f"Raw KB path: {os.path.abspath(phase_kb_path)}")
    print(f"ChromaDB persistence path: {os.path.abspath(persist_directory)}")

    # Ensure the raw KB directory exists for the phase
    if not os.path.exists(phase_kb_path):
        print(f"Directory for {phase} phase does not exist: {os.path.abspath(phase_kb_path)}")
        print(f"Please create it and populate with documents for the '{phase}' phase.")
        return None

    documents = load_documents_from_dir(phase_kb_path)
    if not documents:
        print(f"No documents found for {phase} phase. Skipping DB creation.")
        return None

    # Text Splitting
    print(f"Splitting {len(documents)} documents for {phase} phase...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Increased overlap slightly
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks for {phase} phase.")

    if not chunks:
        print(f"No chunks generated for {phase} phase. Check document content or splitter settings.")
        return None

    # Ollama Embeddings
    print(f"Initializing Ollama embeddings with model: {EMBEDDING_MODEL} for {phase} phase...")
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error initializing OllamaEmbeddings for {phase} phase. Is '{EMBEDDING_MODEL}' pulled and Ollama running? Error: {e}")
        print("Please run 'ollama pull nomic-embed-text' and ensure Ollama server is running.")
        return None

    # Ensure ChromaDB persist directory exists
    os.makedirs(persist_directory, exist_ok=True)

    # Create/Load ChromaDB
    print(f"Creating/Updating vector database for {phase} phase...")
    try:
        # Using .from_documents will create the DB if it doesn't exist or add to it
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        # Chroma automatically persists when using persist_directory in .from_documents,
        # but calling .persist() explicitly ensures data is flushed, especially for updates.
        vectordb.persist()
        print(f"Vector database successfully created/updated for {phase} phase in {os.path.abspath(persist_directory)}")
        return persist_directory
    except Exception as e:
        print(f"Error creating/updating vector database for {phase} phase: {e}")
        return None

# Main Execution Block
if __name__ == "__main__":
    print("--- Starting Multi-Phase Knowledge Base Builder ---")
    print(f"Ensuring base knowledge directory: {os.path.abspath(BASE_KB_DIR)}")
    os.makedirs(BASE_KB_DIR, exist_ok=True)
    print(f"Ensuring base ChromaDB directory: {os.path.abspath(BASE_CHROMA_DB_DIR)}")
    os.makedirs(BASE_CHROMA_DB_DIR, exist_ok=True)

    # Pull Ollama models reminder
    print("\n--- IMPORTANT: Ensure Ollama is running and models are pulled ---")
    print(f"Run 'ollama pull {EMBEDDING_MODEL}' (e.g., 'ollama pull nomic-embed-text')")
    print(f"And 'ollama pull {LLM_MODEL}' (e.g., 'ollama pull llama3')")
    print("-----------------------------------------------------------------\n")


    # Define all SDLC phases and a common base for general security knowledge
    # Using consistent casing (lowercase with underscores) is good practice.
    phases = [
        #"Requirement_phase",
        "Design_phase",
       # "development_phase",
        "testing_phase",
        "Deployment_phase",
        #"common_base" # For general security concepts applicable across phases
    ]

    phase_vdb_paths: Dict[str, Optional[str]] = {}

    for phase in phases:
        vdb_path = create_vector_db(phase)
        phase_vdb_paths[phase] = vdb_path # Store the path, even if it's None (meaning creation failed)

    print("\n--- All vector database creation attempts completed ---")
    print("Summary of created vector DB paths:")
    for phase, path in phase_vdb_paths.items():
        status = "SUCCESS" if path else "FAILED"
        print(f"  {phase}: {os.path.abspath(path) if path else 'N/A'} [{status}]")

    print("\nNext Step: Build your LangChain/LangGraph agents using these paths and your LLM.")