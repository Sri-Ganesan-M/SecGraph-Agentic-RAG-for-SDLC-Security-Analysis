import os
from typing import Dict, Optional, List
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.vectorstores import Chroma

os.environ["GOOGLE_API_KEY"] = "AIzaSyBKHcf5iD2qPu00TNe5WMlzc-1vDZkkyF8"

# Base directory for the raw knowledge base files
BASE_KB_DIR = "/Users/sriganesan/DATA/Software_Security/knowledge_base"

# Base directory for the ChromaDB persistent storage
BASE_CHROMA_DB_DIR = "/Users/sriganesan/DATA/Software_Security/Gemini_api__based/chroma_db"

# Google's embedding model.
EMBEDDING_MODEL = "models/embedding-001"

# Google's LLM model name
LLM_MODEL = "gemini-pro"

# Function to load documents from a directory supporting multiple types
def load_documents_from_dir(folder_path: str) -> List[any]:
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
                continue 

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
    Creates a Chroma vector database for a given SDLC phase using GoogleGenerativeAIEmbeddings.

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

    # Google Generative AI Embeddings
    print(f"Initializing Google Generative AI embeddings with model: {EMBEDDING_MODEL} for {phase} phase...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error initializing GoogleGenerativeAIEmbeddings for {phase} phase. Is GOOGLE_API_KEY set? Error: {e}")
        print("Please ensure your GOOGLE_API_KEY environment variable is set correctly.")
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

if __name__ == "__main__":
    print("--- Starting Multi-Phase Knowledge Base Builder ---")
    print(f"Ensuring base knowledge directory: {os.path.abspath(BASE_KB_DIR)}")
    os.makedirs(BASE_KB_DIR, exist_ok=True)
    print(f"Ensuring base ChromaDB directory: {os.path.abspath(BASE_CHROMA_DB_DIR)}")
    os.makedirs(BASE_CHROMA_DB_DIR, exist_ok=True)

    phases = [
        "Requirement_phase", # Corrected to lowercase and underscore
        "Design_phase",       # Corrected to lowercase and underscore
        "development_phase",  # Already consistent
        "testing_phase",
        "Deployment_phase",   # Corrected to lowercase and underscore
        "common_base"         # Already consistent
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