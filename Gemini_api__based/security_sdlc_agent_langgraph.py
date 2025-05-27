import os
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.agents import AgentFinish, AgentAction
from langchain.agents import Tool

from langgraph.graph import StateGraph, END

# Import tools and config
from security_agnents import tools, EMBEDDING_MODEL, LLM_MODEL # Corrected from security_agnents

# Google API Key setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyBKHcf5iD2qPu00TNe5WMlzc-1vDZkkyF8" # Your hardcoded key

# LLM setup
AGENT_LLM = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0) # Keep temperature at 0 for deterministic tool calling

# --- Agent Prompt Template (MODIFIED) ---
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Application Security Engineer responsible for advising on security throughout the Software Development Lifecycle (SDLC).
You have access to specialized tools for analyzing security at different phases: Requirements, Design, Development, Testing, and Deployment.

Your goal is to provide concise, actionable, and context-aware security advice.

**Instructions for Tool Usage:**
* **Always prioritize using a specific tool** if the user's query clearly relates to one of the SDLC phases (Requirements, Design, Development, Testing, Deployment).
* **Carefully extract all necessary information** from the user's input to provide as arguments to the selected tool.
* **If the user asks for analysis of a document, code, or configuration, use the appropriate tool.**

**Example Scenarios:**
* If the user provides a "user story" or "requirement", use `RequirementsSecurityAnalyzer`.
* If the user provides a "design document" or "architecture description", use `DesignSecurityAnalyzer`.
* If the user provides "code" (e.g., Python, Java, JavaScript) or "pseudocode", use `DevelopmentSecurityAnalyzer`.
* If the user provides a "test plan" or "test results", use `TestingSecurityAnalyzer`.
* If the user provides "deployment configuration" (e.g., AWS, Kubernetes, Docker), use `DeploymentSecurityAnalyzer`.

**General Questions:**
* If the question is general (e.g., "What is SQL injection?", "Tell me about OWASP Top 10"), answer it directly without using a tool.

**Clarification:**
* If the input is ambiguous or you need more details to use a tool, ask clarifying questions.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Define state
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Optional[AgentAction]
    tool_output: Optional[str]

# Agent runner
runnable_agent = agent_prompt | AGENT_LLM.bind_tools(tools) | ToolsAgentOutputParser()

def run_agent(state: AgentState):
    print("---NODE: run_agent (LLM decision)---")
    scratchpad = [msg for msg in state.get("chat_history", []) if isinstance(msg, (ToolMessage, AgentAction))]
    agent_outcome = runnable_agent.invoke({
        "input": state["input"],
        "chat_history": state.get("chat_history", []),
        "agent_scratchpad": scratchpad
    })

    new_chat_history = state.get("chat_history", [])[:]
    if isinstance(agent_outcome, AgentFinish):
        print("Agent decided to finish directly.")
        new_chat_history.append(AIMessage(content=agent_outcome.return_values["output"]))
    elif isinstance(agent_outcome, AgentAction):
        print("Agent decided to call a tool.")
        new_chat_history.append(agent_outcome)

    return {
        "agent_outcome": agent_outcome,
        "chat_history": new_chat_history
    }

def execute_tools(state: AgentState):
    print("---NODE: execute_tools (Manual Tool Execution)---")
    agent_action = state["agent_outcome"]
    print(f"Type of agent_action in execute_tools: {type(agent_action)}")

    if isinstance(agent_action, AgentAction):
        tool_name = agent_action.tool
        tool_input = agent_action.tool_input

        selected_tool_func = None
        for t in tools:
            if t.name == tool_name:
                selected_tool_func = t.func
                break
        
        tool_output_content = ""
        if selected_tool_func:
            try:
                if isinstance(tool_input, dict):
                    tool_output_content = selected_tool_func(**tool_input)
                else:
                    tool_output_content = selected_tool_func(tool_input)
            except Exception as e:
                tool_output_content = f"Error executing tool '{tool_name}' with input '{tool_input}': {e}"
        else:
            tool_output_content = f"Tool '{tool_name}' not found in available tools."


        new_chat_history = state.get("chat_history", [])[:] + [
            ToolMessage(content=str(tool_output_content), name=tool_name)
        ]

        return {
            "tool_output": str(tool_output_content),
            "chat_history": new_chat_history
        }
    else:
        print("WARNING: execute_tools received a non-AgentAction outcome.")
        return {
            "tool_output": "Error: Expected a tool call, but received a direct answer or unexpected type.",
            "chat_history": state.get("chat_history", [])
        }

def route_agent_decision(state: AgentState):
    print("---ROUTER: route_agent_decision---")


    if isinstance(state["agent_outcome"], AgentAction):
        print("Decision: Call Tool")
        return "call_tool"
    else:
        print("Decision: Agent has final answer (END)")
        return "end_conversation"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent_brain", run_agent)
workflow.add_node("tool_executor", execute_tools)

workflow.set_entry_point("agent_brain")

workflow.add_conditional_edges("agent_brain", route_agent_decision, {
    "call_tool": "tool_executor",
    "end_conversation": END
})
workflow.add_edge("tool_executor", END)

app = workflow.compile()

# CLI runner
if __name__ == "__main__":
    print("--- SDLC Security Agent (LangGraph Enabled) ---")
    print("Type your security-related questions or inputs (e.g., a requirement, code, design description).")
    print("Type 'exit' to quit.\n")

    print("Make sure your GOOGLE_API_KEY environment variable is set (or hardcoded as you chose).")
    print("Also ensure ALL your KBs are built with Google embeddings and paths are correct (e.g., in `security_agents.py`).\n")

    chat_history: List[BaseMessage] = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            print("Please enter a query.")
            continue

        try:
            final_state = app.invoke({"input": user_input, "chat_history": chat_history})

            agent_response = "I couldn't process that request effectively or the agent did not provide a final answer."

            if final_state:
                if final_state.get("chat_history") and isinstance(final_state["chat_history"], list) and \
                   final_state["chat_history"] and isinstance(final_state["chat_history"][-1], AIMessage):
                    agent_response = final_state["chat_history"][-1].content
                elif final_state.get("tool_output"):
                    agent_response = final_state["tool_output"]
            
            print(f"Agent: {agent_response}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))

        except Exception as e:
            print(f"An error occurred while processing your request: {e}")
            print("Please try rephrasing your query or check agent logs (if verbose is True).")