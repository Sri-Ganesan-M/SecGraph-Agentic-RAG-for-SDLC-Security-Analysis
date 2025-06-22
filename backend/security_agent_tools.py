import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# --- Configuration ---
# IMPORTANT: Update this path to match where you built your ChromaDBs
BASE_CHROMA_DB_DIR = "/Users/sriganesan/DATA/DevSecOps/Knowledge_Base_Builder/chroma_db"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash-latest"

# --- LLM and Embeddings ---
llm_for_tools = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# --- Knowledge Base Paths ---
PHASE_PATHS = {
    "requirement": os.path.join(BASE_CHROMA_DB_DIR, "requirement_phase_security_kb"),
    "design": os.path.join(BASE_CHROMA_DB_DIR, "design_phase_security_kb"),
    "development": os.path.join(BASE_CHROMA_DB_DIR, "development_phase_security_kb"),
    "testing": os.path.join(BASE_CHROMA_DB_DIR, "testing_phase_security_kb"),
    "deployment": os.path.join(BASE_CHROMA_DB_DIR, "deployment_phase_security_kb"),
    "common": os.path.join(BASE_CHROMA_DB_DIR, "common_base_security_kb"),
}

# --- Retriever Loading ---
def load_retriever(db_path: str):
    if not os.path.exists(db_path):
        print(f"Warning: Vector DB not found at {db_path}.")
        return None
    vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3})

retrievers = {phase: load_retriever(path) for phase, path in PHASE_PATHS.items()}
common_retriever = retrievers.get("common")

def combined_retriever_for_phase(phase_retriever: Optional[any], query: str) -> List[any]:
    phase_docs = phase_retriever.invoke(query) if phase_retriever else []
    common_docs = common_retriever.invoke(query) if common_retriever else []
    return list({doc.page_content: doc for doc in phase_docs + common_docs}.values())

# --- Tool Functions (MCP-Aware) ---

def analyze_requirement_security(requirement_text: str, context_summary: str) -> str:
    """Analyzes a software requirement for security considerations."""
    req_retriever = retrievers.get("requirement")
    if not req_retriever: return "Error: Requirement knowledge base not loaded."

    prompt = ChatPromptTemplate.from_template(
        """You are an expert in secure software requirements.

        **Previous Analysis Context for this File:**
        {context_summary}

        **Your Task:**
        Review the user story/requirement below. Identify potential security risks, missing security requirements, or abuse cases.
        Provide actionable suggestions to make the requirement more secure.

        **Requirement to Analyze:**
        ```
        {requirement_text}
        ```

        **Relevant Security Practices from Knowledge Base:**
        {context}
        """
    )
    
    chain = (
        {
            "context": itemgetter("requirement_text") | RunnableLambda(lambda q: combined_retriever_for_phase(req_retriever, q)),
            "requirement_text": itemgetter("requirement_text"),
            "context_summary": itemgetter("context_summary"),
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )
    return chain.invoke({
        "requirement_text": requirement_text,
        "context_summary": context_summary
    })

def analyze_design_security(design_description: str, technology_stack: str, context_summary: str) -> str:
    """Analyzes a system design for security flaws."""
    design_retriever = retrievers.get("design")
    if not design_retriever: return "Error: Design knowledge base not loaded."

    prompt = ChatPromptTemplate.from_template(
        """You are an expert in secure system design and architecture, with expertise in {technology_stack}.

        **Previous Analysis Context for this File:**
        {context_summary}

        **Your Task:**
        Review the system design description below. Look for insecure design patterns, threat modeling gaps (e.g., STRIDE), trust boundary issues, and missing security controls. Provide specific, actionable advice to harden the design.

        **Design to Analyze (Technology: {technology_stack}):**
        ```
        {design_description}
        ```

        **Relevant Security Practices from Knowledge Base:**
        {context}
        """
    )

    chain = (
        {
            "context": itemgetter("design_description") | RunnableLambda(lambda q: combined_retriever_for_phase(design_retriever, q)),
            "design_description": itemgetter("design_description"),
            "technology_stack": itemgetter("technology_stack"),
            "context_summary": itemgetter("context_summary"),
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )
    return chain.invoke({
        "design_description": design_description,
        "technology_stack": technology_stack,
        "context_summary": context_summary
    })

def analyze_development_security(code_snippet: str, language: str, context_summary: str) -> str:
    """Analyzes a code snippet for security vulnerabilities."""
    dev_retriever = retrievers.get("development")
    if not dev_retriever: return "Error: Development knowledge base not loaded."

    prompt = ChatPromptTemplate.from_template(
        """You are an expert Application Security Engineer specializing in {language}.

        **Previous Analysis Context for this File:**
        {context_summary}

        **Your Task:**
        Review the following code snippet. Based on the previous context and the full code provided, give a security analysis.
        - If a previously mentioned issue is now fixed, acknowledge it.
        - If new issues are present, detail them clearly with a description, risk, and suggested fix.
        - If no issues are found, state that the code looks secure.

        **Code to Analyze:**
        ```
        {code_snippet}
        ```

        **Relevant Security Practices from Knowledge Base:**
        {context}
        """
    )

    chain = (
        {
            "context": itemgetter("code_snippet") | RunnableLambda(lambda q: combined_retriever_for_phase(dev_retriever, q)),
            "code_snippet": itemgetter("code_snippet"),
            "language": itemgetter("language"),
            "context_summary": itemgetter("context_summary"),
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )
    return chain.invoke({
        "code_snippet": code_snippet,
        "language": language,
        "context_summary": context_summary
    })

def analyze_testing_security(test_plan_or_result: str, testing_type: str, context_summary: str) -> str:
    """Analyzes a security test plan or test results for gaps."""
    test_retriever = retrievers.get("testing")
    if not test_retriever: return "Error: Testing knowledge base not loaded."

    prompt = ChatPromptTemplate.from_template(
        """You are an expert in security testing and quality assurance, with expertise in {testing_type}.

        **Previous Analysis Context for this File:**
        {context_summary}

        **Your Task:**
        Review the security test plan or results below. Suggest additional test cases, recommend relevant security testing methodologies (e.g., OWASP Testing Guide), or provide guidance on interpreting the findings and prioritizing remediation efforts.

        **Test Plan/Results to Analyze (Type: {testing_type}):**
        ```
        {test_plan_or_result}
        ```

        **Relevant Security Practices from Knowledge Base:**
        {context}
        """
    )

    chain = (
        {
            "context": itemgetter("test_plan_or_result") | RunnableLambda(lambda q: combined_retriever_for_phase(test_retriever, q)),
            "test_plan_or_result": itemgetter("test_plan_or_result"),
            "testing_type": itemgetter("testing_type"),
            "context_summary": itemgetter("context_summary"),
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )
    return chain.invoke({
        "test_plan_or_result": test_plan_or_result,
        "testing_type": testing_type,
        "context_summary": context_summary
    })

def analyze_deployment_security(deployment_config: str, environment: str, context_summary: str) -> str:
    """Analyzes deployment configurations or IaC for misconfigurations."""
    deploy_retriever = retrievers.get("deployment")
    if not deploy_retriever: return "Error: Deployment knowledge base not loaded."

    prompt = ChatPromptTemplate.from_template(
        """You are an expert in secure deployment, infrastructure, and cloud security, with expertise in {environment}.

        **Previous Analysis Context for this File:**
        {context_summary}

        **Your Task:**
        Review the deployment configuration or Infrastructure-as-Code (IaC) file below. Identify security misconfigurations, insecure defaults, excessive permissions, or compliance gaps (e.g., CIS Benchmarks). Provide clear, actionable steps to harden the configuration.

        **Configuration to Analyze (Environment: {environment}):**
        ```
        {deployment_config}
        ```

        **Relevant Security Practices from Knowledge Base:**
        {context}
        """
    )

    chain = (
        {
            "context": itemgetter("deployment_config") | RunnableLambda(lambda q: combined_retriever_for_phase(deploy_retriever, q)),
            "deployment_config": itemgetter("deployment_config"),
            "environment": itemgetter("environment"),
            "context_summary": itemgetter("context_summary"),
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )
    return chain.invoke({
        "deployment_config": deployment_config,
        "environment": environment,
        "context_summary": context_summary
    })


# --- General Chat Assistant Tool ---

def get_all_retrievers() -> List[any]:
    """Returns a list of all successfully loaded retrievers."""
    return [r for r in retrievers.values() if r is not None]

def general_purpose_retriever(query: str) -> List[any]:
    """
    Invokes all available retrievers and combines the results to answer general questions.
    This allows the chat assistant to draw from the entire knowledge base.
    """
    all_docs = []
    for retriever in get_all_retrievers():
        try:
            all_docs.extend(retriever.invoke(query))
        except Exception as e:
            print(f"Error invoking a retriever: {e}")
    
    # Deduplicate results based on page content
    return list({doc.page_content: doc for doc in all_docs}.values())

def chat_with_security_assistant(question: str, chat_history: str) -> str:
    """Answers general security questions by searching all available knowledge bases."""
    
    if not get_all_retrievers():
        return "Error: No knowledge bases are loaded, cannot answer questions."

    prompt = ChatPromptTemplate.from_template(
        """You are a friendly and helpful AI Security Assistant. Your role is to answer questions about software security, vulnerabilities, or best practices.

        Use the provided chat history to understand the context of the conversation.
        Use the retrieved knowledge base context to provide accurate, detailed, and helpful answers.

        **Chat History:**
        {chat_history}

        **Retrieved Knowledge Base Context:**
        {context}

        **User's Question:**
        {question}

        **Your Answer:**
        """
    )

    chain = (
        {
            "context": itemgetter("question") | RunnableLambda(general_purpose_retriever),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )

    return chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })