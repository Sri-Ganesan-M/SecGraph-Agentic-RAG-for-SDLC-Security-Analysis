import os
# --- Insert your API Key directly here (FOR TESTING ONLY - NOT RECOMMENDED FOR PRODUCTION) ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyBKHcf5iD2qPu00TNe5WMlzc-1vDZkkyF8"
# -----------------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Import the tools defined in your security_agents.py
from security_agnents import tools, EMBEDDING_MODEL, LLM_MODEL # Ensure LLM_MODEL is imported for consistency
# --- Configuration (Consistent with KB Builder) ---
BASE_CHROMA_DB_DIR = "chroma_db"
# --- CHANGE START ---
EMBEDDING_MODEL = "models/embedding-001" # Google's embedding model
LLM_MODEL = "gemini-2.0-flash" # Google's powerful chat model
# --- Configuration for the Agent's Brain LLM ---
AGENT_LLM = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0) # Use lower temperature for more deterministic tool selection

# --- Define the Agent's System Prompt ---
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert Application Security Engineer responsible for advising on security throughout the Software Development Lifecycle (SDLC).
         You have access to specialized tools for analyzing security at different phases: Requirements, Design, Development, Testing, and Deployment.
         
         Your goal is to provide concise, actionable, and context-aware security advice.
         
         When a user provides input:
         1. Carefully determine which SDLC phase the input relates to.
         2. Select the MOST appropriate specialized security analysis tool from your available tools.
         3. Provide the exact input required by the selected tool, ensuring all necessary parameters are included (e.g., code snippet AND language for DevelopmentSecurityAnalyzer).
         4. If the input is ambiguous or you need more information to pick a tool or form its input, ask clarifying questions.
         5. If no specific security analysis tool seems applicable, provide general security guidance or explain why no specific tool was used.
         """
        ),
        MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's internal thought process
    ]
)

# --- Create the Agent ---
security_agent = create_tool_calling_agent(AGENT_LLM, tools, agent_prompt)

# --- Create the Agent Executor ---
agent_executor = AgentExecutor(
    agent=security_agent,
    tools=tools,
    verbose=True, # Set to True to see the agent's thought process (useful for debugging)
    handle_parsing_errors=True, # Helps recover from LLM outputting incorrect tool calls
)

# --- Main Interaction Loop ---
if __name__ == "__main__":
    print("--- SDLC Security Agent ---")
    print("Type your security-related questions or inputs (e.g., a requirement, code, design description).")
    print("Type 'exit' to quit.\n")

    # --- IMPORTANT: Ensure Google API Key is set (or hardcoded as you chose) ---
    print("Make sure your GOOGLE_API_KEY environment variable is set (or hardcoded for testing as you chose).")
    print(f"Also ensure ALL your KBs are built with Google embeddings and paths are correct (e.g., in `security_agents.py`).\n")


    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            print("Please enter a query.")
            continue

        try:
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            
            agent_response = result.get("output", "No specific output from agent.")
            print(f"Agent: {agent_response}")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))

        except Exception as e:
            print(f"An error occurred while processing your request: {e}")
            print("Please try rephrasing your query or check agent logs (if verbose is True).")