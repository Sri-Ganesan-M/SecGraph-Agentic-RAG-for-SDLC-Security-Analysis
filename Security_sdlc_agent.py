import os
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Import the tools defined in your security_analysis_tools.py
from security_analysis_agents import tools, LLM_MODEL # Also import LLM_MODEL for consistency
EMBEDDING_MODEL = "nomic-embed-text" # Must be pulled via `ollama pull nomic-embed-text`
LLM_MODEL = "mistral" # Must be pulled via `ollama pull codellama:7b`
# --- Configuration for the Agent's Brain LLM ---
# This LLM will be used by the agent to reason and select tools.
# It should be a robust model capable of reasoning and instruction following.
# Using the same LLM_MODEL from tools for consistency, ensure it's pulled.
AGENT_LLM = ChatOllama(model=LLM_MODEL, temperature=0) # Use lower temperature for more deterministic tool selection

# --- Define the Agent's System Prompt ---
# This prompt guides the LLM on its role, capabilities (tools), and how to behave.
# It's crucial for effective tool selection.
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert Application Security Engineer responsible for advising on security throughout the Software Development Lifecycle (SDLC).
         You have access to specialized tools for analyzing security at different phases: Requirements, Design, Development, Testing, and Deployment.
         
         Your goal is to provide concise, actionable, and context-aware security advice.
         
         When a user provides input:
         1.  Carefully determine which SDLC phase the input relates to.
         2.  Select the MOST appropriate specialized security analysis tool from your available tools.
         3.  Provide the exact input required by the selected tool, ensuring all necessary parameters are included (e.g., code snippet AND language for DevelopmentSecurityAnalyzer).
         4.  If the input is ambiguous or you need more information to pick a tool or form its input, ask clarifying questions.
         5.  If no specific security analysis tool seems applicable, provide general security guidance or explain why no specific tool was used.
         """
        ),
        MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's internal thought process
    ]
)

# --- Create the Agent ---
# create_tool_calling_agent automatically injects tool descriptions and handles tool calls.
security_agent = create_tool_calling_agent(AGENT_LLM, tools, agent_prompt)

# --- Create the Agent Executor ---
# The AgentExecutor is responsible for running the agent, managing its loop,
# executing tools, and handling observations.
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

    # --- IMPORTANT: Ensure Ollama is running and models are pulled ---
    print(f"Make sure Ollama is running and you have pulled '{EMBEDDING_MODEL}' (used by tools) and '{LLM_MODEL}' (used by agent and tools).")
    print(f"e.g., `ollama pull nomic-embed-text` and `ollama pull {LLM_MODEL}`\n")
    print("Also ensure ALL your KBs are built using 'build_all_kbs.py' or an updated 'build_selective_kbs.py'.\n")


    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            print("Please enter a query.")
            continue

        try:
            # Invoke the agent executor with the current input and chat history
            # The agent_executor will decide which tool to call based on `user_input`
            # and the `agent_prompt`.
            result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
            
            agent_response = result.get("output", "No specific output from agent.")
            print(f"Agent: {agent_response}")
            
            # Update chat history for conversational memory
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))

        except Exception as e:
            print(f"An error occurred while processing your request: {e}")
            print("Please try rephrasing your query or check agent logs (if verbose is True).")