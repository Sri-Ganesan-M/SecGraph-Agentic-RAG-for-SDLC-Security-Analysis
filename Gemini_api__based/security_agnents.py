import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.agents import Tool 
os.environ["GOOGLE_API_KEY"] = ""
BASE_CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "models/embedding-001" 
LLM_MODEL = "gemini-2.0-flash" 

REQUIREMENTS_KB_PATH = os.path.join(BASE_CHROMA_DB_DIR, "Requirement_phase_security_kb")
DESIGN_KB_PATH = os.path.join(BASE_CHROMA_DB_DIR, "design_phase_security_kb")
DEVELOPMENT_KB_PATH = os.path.join(BASE_CHROMA_DB_DIR, "development_phase_security_kb")
TESTING_KB_PATH = os.path.join(BASE_CHROMA_DB_DIR, "testing_phase_security_kb")
DEPLOYMENT_KB_PATH = os.path.join(BASE_CHROMA_DB_DIR, "deployment_phase_security_kb")
COMMON_KB_PATH = os.path.join(BASE_CHROMA_DB_DIR, "common_base_security_kb")


def load_and_configure_retriever(db_path: str, embedding_model_name: str):
    """Loads a ChromaDB and configures its retriever."""
    if not os.path.exists(db_path):
        print(f"Error: Vector DB not found at {db_path}. Please ensure it was built.")
        return None
    try:
        # --- CHANGE START ---
        embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        # --- CHANGE END ---
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        print(f"Successfully loaded ChromaDB from {db_path}")
        return vectordb.as_retriever(search_kwargs={"k": 4}) # Retrieve top 4 relevant chunks
    except Exception as e:
        print(f"Error loading ChromaDB from {db_path}: {e}")
        return None


llm_for_tools = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2) 


req_retriever = load_and_configure_retriever(REQUIREMENTS_KB_PATH, EMBEDDING_MODEL)
design_retriever = load_and_configure_retriever(DESIGN_KB_PATH, EMBEDDING_MODEL)
dev_retriever = load_and_configure_retriever(DEVELOPMENT_KB_PATH, EMBEDDING_MODEL)
test_retriever = load_and_configure_retriever(TESTING_KB_PATH, EMBEDDING_MODEL)
deploy_retriever = load_and_configure_retriever(DEPLOYMENT_KB_PATH, EMBEDDING_MODEL)
common_retriever = load_and_configure_retriever(COMMON_KB_PATH, EMBEDDING_MODEL)

# Define combined retriever for specific phases (each tool might use its own phase KB + common KB)
def combined_retriever_for_phase(phase_retriever: Optional[any], query: str) -> List[any]:
    """Combines retrieval from a specific phase KB and the common KB."""
    phase_docs = phase_retriever.invoke(query) if phase_retriever else []
    common_docs = common_retriever.invoke(query) if common_retriever else []
    all_docs = {doc.page_content: doc for doc in phase_docs + common_docs} # Deduplicate
    return list(all_docs.values())

# --- 1. Requirements Phase Security Tool ---
def get_requirements_security_suggestions(requirement_text: str) -> str:
    """
    Analyzes a given software requirement text and provides security suggestions
    by identifying potential security considerations or missing security requirements.
    Uses requirements_phase_security_kb and common_base_security_kb.
    """
    if not (req_retriever or common_retriever):
        return "Error: Requirements or Common knowledge bases not loaded. Cannot provide suggestions."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """You are an expert Application Security Engineer specializing in the Requirements phase.
             Your task is to review user stories or functional requirements and identify potential security considerations or missing security requirements.
             Refer to the provided context from security best practices, standards, and common vulnerabilities related to requirements.
             For each potential security risk or missing aspect, provide:
             1. A clear description of the security concern.
             2. Why it's a concern (briefly).
             3. A specific suggestion for how to address it in the requirements or design phase.
             4. Reference the context if applicable (e.g., "As per OWASP ASVS...").
             If no obvious security concerns are found, state that and suggest general best practices like thorough threat modeling and security reviews.
             Be concise, actionable, and prioritize the most significant concerns first."""
            ),
            ("human", "Here is a user requirement to analyze:\n\n{requirement}\n\nRelevant context:\n{context}"),
        ]
    )

    rag_chain = (
        {
            "context": itemgetter("requirement_text") | RunnableLambda(lambda x: combined_retriever_for_phase(req_retriever, x)),
            "requirement": itemgetter("requirement_text"),
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )
    print(f"Invoking Requirements Security Tool for: {requirement_text[:50]}...")
    try:
        response = rag_chain.invoke({"requirement_text": requirement_text})
        return response
    except Exception as e:
        return f"An error occurred while processing the requirement: {e}"

# --- 2. Design Phase Security Tool ---
def analyze_design_security(design_description: str, technology_stack: str = "general") -> str:
    """
    Analyzes a system's design description, architectural diagrams, or data flow diagrams
    for security vulnerabilities, insecure design patterns, and missing security controls.
    Suggests secure design principles and patterns.
    Specify the technology stack if known (e.g., "Microservices", "Cloud Native", "Web Application", "general").
    Uses design_phase_security_kb and common_base_security_kb.
    """
    if not (design_retriever or common_retriever):
        return "Error: Design or Common knowledge bases not loaded. Cannot analyze security."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"""You are an expert Application Security Engineer specializing in the Design phase, with expertise in {technology_stack} security.
             Your task is to review the provided system design, architectural diagrams, or data flow descriptions for security vulnerabilities,
             insecure design patterns, and missing security controls.
             Refer to the provided context from secure design patterns, threat modeling methodologies (e.g., STRIDE), and architectural security principles.
             Analyze the input carefully. For each identified design flaw or missing aspect, provide:
             1. A clear description of the security issue.
             2. Why it's a concern (briefly).
             3. A specific suggestion for a secure design alternative or control, referencing secure design patterns if applicable.
             4. Reference the context if applicable (e.g., "As per STRIDE threat modeling...").
             If no obvious security concerns are found, state that and suggest general secure design review best practices, like performing a dedicated threat modeling session.
             Be concise, actionable, and prioritize the most significant concerns first."""
            ),
            ("human", "Here is the design description to analyze (Technology Stack: {technology_stack}):\n\n```\n{design_description}\n```\n\nRelevant context:\n{context}"),
        ]
    )

    rag_chain = (
        {
            "context": itemgetter("design_description") | RunnableLambda(lambda x: combined_retriever_for_phase(design_retriever, x)),
            "design_description": itemgetter("design_description"),
            "technology_stack": itemgetter("technology_stack")
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )

    print(f"Invoking Design Security Tool for: {design_description[:50]}... (Tech: {technology_stack})")
    try:
        response = rag_chain.invoke({"design_description": design_description, "technology_stack": technology_stack})
        return response
    except Exception as e:
        return f"An error occurred while processing the design: {e}"

# --- 3. Development Phase Security Tool (Existing) ---
def analyze_development_security(code_or_design_snippet: str, language: str = "general") -> str:
    """
    Analyzes a code snippet, pseudocode, or detailed design description for security vulnerabilities.
    Suggests potential fixes or secure coding practices.
    Specify the programming language (e.g., "Python", "Java", "JavaScript", "Go", "general").
    Uses development_phase_security_kb and common_base_security_kb.
    """
    if not (dev_retriever or common_retriever):
        return "Error: Development or Common knowledge bases not loaded. Cannot analyze security."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"""You are an expert Application Security Engineer specializing in the Development phase, with expertise in {language} security.
             Your task is to review the provided code snippet, pseudocode, or detailed design description for security vulnerabilities or insecure practices.
             Refer to the provided context from secure coding guidelines, common vulnerabilities (like OWASP Top 10), and secure design patterns.
             Analyze the input carefully. For each identified vulnerability or insecure practice, provide:
             1. A clear description of the security issue (e.g., "Potential SQL Injection vulnerability").
             2. Why it's a concern (briefly).
             3. A specific suggestion for how to fix it or a secure alternative, ideally with code examples if applicable.
             4. Reference the context if applicable (e.g., "As per OWASP Secure Coding Practices...").
             If no obvious security concerns are found, state that and suggest general secure development best practices for {language}.
             Be concise, actionable, and prioritize the most significant concerns first.
             If the input is pseudocode or design, focus on logical or architectural security flaws."""
            ),
            ("human", "Here is the code/design snippet to analyze (Language: {language}):\n\n```\n{snippet}\n```\n\nRelevant context:\n{context}"),
        ]
    )

    rag_chain = (
        {
            "context": itemgetter("code_or_design_snippet") | RunnableLambda(lambda x: combined_retriever_for_phase(dev_retriever, x)),
            "snippet": itemgetter("code_or_design_snippet"),
            "language": itemgetter("language") # Pass language through
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )

    print(f"Invoking Development Security Tool for: {code_or_design_snippet[:50]}... (Language: {language})")
    try:
        response = rag_chain.invoke({"code_or_design_snippet": code_or_design_snippet, "language": language})
        return response
    except Exception as e:
        return f"An error occurred while processing the development snippet: {e}"

# --- 4. Testing Phase Security Tool ---
def get_testing_security_suggestions(test_plan_or_result: str, testing_type: str = "general") -> str:
    """
    Analyzes a security test plan, test cases, or test results.
    Suggests additional security test cases, methodologies, or interpretation of results.
    Specify the testing type (e.g., "Penetration Test", "SAST", "DAST", "Fuzzing", "general").
    Uses testing_phase_security_kb and common_base_security_kb.
    """
    if not (test_retriever or common_retriever):
        return "Error: Testing or Common knowledge bases not loaded. Cannot analyze security."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"""You are an expert Application Security Engineer specializing in the Testing phase, with expertise in {testing_type} security testing.
             Your task is to review the provided security test plan, specific test cases, or security test results.
             Refer to the provided context from security testing methodologies (e.g., OWASP Testing Guide), common vulnerability testing scenarios, and security test tool outputs.
             Analyze the input carefully. Provide:
             1. Suggestions for additional security test cases that might be missing.
             2. Recommendations for relevant security testing methodologies or tools to apply.
             3. Guidance on interpreting security test results or prioritizing findings.
             4. Reference the context if applicable (e.g., "As per OWASP Testing Guide...").
             If the provided input seems comprehensive, state that and suggest general best practices for continuous security testing.
             Be concise, actionable, and focus on improving test coverage and effectiveness."""
            ),
            ("human", "Here is the security test plan/result to analyze (Testing Type: {testing_type}):\n\n```\n{test_plan_or_result}\n```\n\nRelevant context:\n{context}"),
        ]
    )

    rag_chain = (
        {
            "context": itemgetter("test_plan_or_result") | RunnableLambda(lambda x: combined_retriever_for_phase(test_retriever, x)),
            "test_plan_or_result": itemgetter("test_plan_or_result"),
            "testing_type": itemgetter("testing_type")
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )

    print(f"Invoking Testing Security Tool for: {test_plan_or_result[:50]}... (Type: {testing_type})")
    try:
        response = rag_chain.invoke({"test_plan_or_result": test_plan_or_result, "testing_type": testing_type})
        return response
    except Exception as e:
        return f"An error occurred while processing the test plan/result: {e}"

# --- 5. Deployment Phase Security Tool ---
def analyze_deployment_security(deployment_config: str, environment: str = "general") -> str:
    """
    Analyzes deployment configurations, infrastructure-as-code (IaC) files,
    container images, or cloud configurations for security misconfigurations,
    insecure defaults, and compliance issues.
    Suggests secure deployment practices and hardening steps.
    Specify the environment (e.g., "AWS", "Kubernetes", "Docker", "On-Premise", "general").
    Uses deployment_phase_security_kb and common_base_security_kb.
    """
    if not (deploy_retriever or common_retriever):
        return "Error: Deployment or Common knowledge bases not loaded. Cannot analyze security."

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"""You are an expert Application Security Engineer specializing in the Deployment phase, with expertise in {environment} security.
             Your task is to review the provided deployment configuration, infrastructure-as-code (IaC), container image definition, or cloud configuration.
             Refer to the provided context from secure deployment checklists, cloud security best practices, container hardening guides, and compliance standards.
             Analyze the input carefully. For each identified misconfiguration, insecure default, or compliance gap, provide:
             1. A clear description of the security issue.
             2. Why it's a concern (briefly).
             3. A specific suggestion for how to fix it or a secure configuration, ideally with example code/config snippets.
             4. Reference the context if applicable (e.g., "As per CIS Benchmark for Kubernetes...").
             If no obvious security concerns are found, state that and suggest general best practices for continuous security monitoring and configuration management.
             Be concise, actionable, and focus on improving the security posture of the deployed environment."""
            ),
            ("human", "Here is the deployment configuration to analyze (Environment: {environment}):\n\n```\n{deployment_config}\n```\n\nRelevant context:\n{context}"),
        ]
    )

    rag_chain = (
        {
            "context": itemgetter("deployment_config") | RunnableLambda(lambda x: combined_retriever_for_phase(deploy_retriever, x)),
            "deployment_config": itemgetter("deployment_config"),
            "environment": itemgetter("environment")
        }
        | prompt
        | llm_for_tools
        | StrOutputParser()
    )

    print(f"Invoking Deployment Security Tool for: {deployment_config[:50]}... (Env: {environment})")
    try:
        response = rag_chain.invoke({"deployment_config": deployment_config, "environment": environment})
        return response
    except Exception as e:
        return f"An error occurred while processing the deployment configuration: {e}"

# --- Register All Tools for LangChain Agent ---
# These are the actual Tool objects that a LangChain agent can use.
# The 'description' is crucial as the LLM uses it to decide which tool to call.
# Make sure the `func` points to the correct function and `description` accurately reflects its purpose.
tools = [
    Tool(
        name="RequirementsSecurityAnalyzer",
        func=get_requirements_security_suggestions,
        description="""Useful for analyzing software requirements or user stories to identify potential security considerations or missing security requirements. 
                       Input should be the full text of the user requirement or user story. Example: 'As a user, I want to log in using my username and password.'""",
    ),
    Tool(
        name="DesignSecurityAnalyzer",
        func=analyze_design_security,
        description="""Useful for analyzing system design descriptions, architectural diagrams, or data flow diagrams for security vulnerabilities, insecure design patterns, and missing controls. 
                       Input should be the design description optionally followed by the technology stack (e.g., 'Microservices', 'Cloud Native', 'Web Application'). 
                       Example input: 'Our system uses microservices architecture with REST APIs. | Microservices'""",
    ),
    Tool(
        name="DevelopmentSecurityAnalyzer",
        func=analyze_development_security,
        description="""Useful for analyzing code snippets, pseudocode, or detailed technical design descriptions for security vulnerabilities and suggesting fixes. 
                       Input should be the code/design snippet followed by the programming language (e.g., 'Python', 'Java', 'JavaScript', 'general'). 
                       Example input: 'def login(username, password): return f\"SELECT * FROM users WHERE user='{username}' AND pass='{password}'\" | Python'""",
    ),
    Tool(
        name="TestingSecurityAnalyzer",
        func=get_testing_security_suggestions,
        description="""Useful for analyzing security test plans, specific test cases, or security test results. It suggests additional security test cases, methodologies, or interpretation of results. 
                       Input should be the test plan/result description optionally followed by the testing type (e.g., 'Penetration Test', 'SAST', 'DAST', 'Fuzzing'). 
                       Example input: 'Our current test plan only includes functional tests. | general'""",
    ),
    Tool(
        name="DeploymentSecurityAnalyzer",
        func=analyze_deployment_security,
        description="""Useful for analyzing deployment configurations, infrastructure-as-code (IaC) files, container images, or cloud configurations for security misconfigurations. 
                       It suggests secure deployment practices and hardening steps. 
                       Input should be the configuration text optionally followed by the environment (e.g., 'AWS', 'Kubernetes', 'Docker'). 
                       Example input: 'Our AWS S3 bucket is public. | AWS'""",
    )
]

# --- Example Usage (for testing this file directly) ---
if __name__ == "__main__":
    print("--- Testing All Security Analysis Tools ---")

    # --- IMPORTANT: Ensure Google API Key is set ---
    print("Make sure your GOOGLE_API_KEY environment variable is set.")
    print("e.g., `export GOOGLE_API_KEY='YOUR_API_KEY'` on Linux/macOS or `$env:GOOGLE_API_KEY='YOUR_API_KEY'` on PowerShell.\n")
    print(f"Also ensure ALL your KBs are built and paths are correct (e.g., '{REQUIREMENTS_KB_PATH}').\n")


    # --- Test Requirements Tool ---
    print("--- Testing RequirementsSecurityAnalyzer ---")
    req_input = """
    As a user, I want to create an account by providing my email and a password. The system should send a welcome email.
    """
    req_suggestion = get_requirements_security_suggestions(req_input.strip())
    print("\nSecurity Suggestions for Requirement:\n", req_suggestion)
    print("\n" + "="*80 + "\n")

