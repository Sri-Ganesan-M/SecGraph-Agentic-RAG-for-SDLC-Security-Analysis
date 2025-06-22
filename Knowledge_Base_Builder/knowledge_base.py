import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader, UnstructuredMarkdownLoader, UnstructuredPDFLoader, UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Load environment variables from a .env file if it exists
load_dotenv()

# IMPORTANT: Set your GOOGLE_API_KEY as an environment variable
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# IMPORTANT: Update these paths to match your system
BASE_KB_DOCS_DIR = "/Users/sriganesan/DATA/Software_Security/knowledge_base"
BASE_CHROMA_DB_DIR = "chroma_db"

EMBEDDING_MODEL = "models/embedding-001"

def load_documents_from_dir(folder_path: str) -> List[any]:
    # (Code from your original file - no changes needed here)
    documents = []
    print(f"Loading documents from: {os.path.abspath(folder_path)}")
    if not os.path.exists(folder_path):
        print(f"Error: Directory not found: {folder_path}. Please create it.")
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
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return documents

def create_vector_db(phase: str) -> Optional[str]:
    # (Code from your original file - no changes needed here)
    phase_kb_path = os.path.join(BASE_KB_DOCS_DIR, phase)
    persist_directory = os.path.join(BASE_CHROMA_DB_DIR, f"{phase}_security_kb")
    print(f"\n--- Processing Phase: {phase} ---")
    if not os.path.exists(phase_kb_path):
        print(f"Directory for {phase} phase does not exist: {os.path.abspath(phase_kb_path)}")
        return None
    documents = load_documents_from_dir(phase_kb_path)
    if not documents:
        print(f"No documents found for {phase} phase. Skipping.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        print(f"No chunks generated for {phase} phase.")
        return None
    print(f"Initializing embeddings for {phase}...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"Creating vector database for {phase}...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector DB for {phase} created in {os.path.abspath(persist_directory)}")
    return persist_directory

if __name__ == "__main__":
    print("--- Starting Multi-Phase Knowledge Base Builder ---")
    os.makedirs(BASE_KB_DOCS_DIR, exist_ok=True)
    os.makedirs(BASE_CHROMA_DB_DIR, exist_ok=True)

    # List of folders inside your BASE_KB_DOCS_DIR
    phases = [
        "requirement_phase",
        "design_phase",
        "development_phase",
        "testing_phase",
        "deployment_phase",
        "common_base"
    ]
    for phase in phases:
        create_vector_db(phase)
    print("\n--- All vector database creation attempts completed ---")