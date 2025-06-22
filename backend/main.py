import os
import json
import hashlib
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

from mcp_database import init_db, get_context, save_context
from security_agent_tools import (
    analyze_development_security,
    analyze_requirement_security,
    analyze_design_security,
    analyze_testing_security,
    analyze_deployment_security,
    chat_with_security_assistant # <-- This is the function we will call
)

# Initialize the MCP database on startup
init_db()

app = FastAPI(title="SDLC Security Agent API")

# --- Pydantic Models ---

class AnalysisRequest(BaseModel):
    file_path: str
    content: str
    language: str = "general"
    technology_stack: Optional[str] = "general"
    testing_type: Optional[str] = "general"
    environment: Optional[str] = "general"

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[str] = "No history yet."


# --- Helper Functions ---
def distill_analysis_for_context(full_analysis: str) -> str:
    """Placeholder for a real summarization agent."""
    return (full_analysis[:300] + '...') if len(full_analysis) > 300 else full_analysis


# --- API Endpoints ---

@app.post("/analyze/{phase}")
async def analyze_phase_endpoint(phase: str, request: AnalysisRequest):
    """
    Main analysis endpoint that uses the Model Context Protocol (MCP).
    """
    tool_function_map = {
        "requirement": analyze_requirement_security,
        "design": analyze_design_security,
        "development": analyze_development_security,
        "testing": analyze_testing_security,
        "deployment": analyze_deployment_security,
    }
    tool_function = tool_function_map.get(phase)
    if not tool_function:
        raise HTTPException(status_code=404, detail=f"Analysis phase '{phase}' not found.")

    previous_context = get_context(request.file_path)
    context_summary = "This is the first time this file is being analyzed."
    if previous_context:
        context_summary = previous_context.get("llm_summary_for_next_run", context_summary)
        new_hash = hashlib.sha256(request.content.encode()).hexdigest()
        if previous_context.get("last_analyzed_hash") == new_hash:
            return {
                "message": "No changes detected since last analysis.",
                "analysis": previous_context.get("last_full_analysis", "")
            }

    print(f"--- Analyzing {request.file_path} for phase: {phase} ---")
    
    try:
        # Pass arguments based on the phase
        if phase == "development":
            args = {"code_snippet": request.content, "language": request.language, "context_summary": context_summary}
        elif phase == "design":
            args = {"design_description": request.content, "technology_stack": request.technology_stack, "context_summary": context_summary}
        elif phase == "testing":
            args = {"test_plan_or_result": request.content, "testing_type": request.testing_type, "context_summary": context_summary}
        elif phase == "deployment":
            args = {"deployment_config": request.content, "environment": request.environment, "context_summary": context_summary}
        else: # Requirement
            args = {"requirement_text": request.content, "context_summary": context_summary}
        
        full_analysis_result = tool_function(**args)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")

    new_summary = distill_analysis_for_context(full_analysis_result)
    new_context_obj = {
        "file_path": request.file_path,
        "last_analyzed_hash": hashlib.sha256(request.content.encode()).hexdigest(),
        "last_analysis_timestamp": datetime.datetime.utcnow().isoformat(),
        "llm_summary_for_next_run": new_summary,
        "last_full_analysis": full_analysis_result
    }
    save_context(request.file_path, new_context_obj)

    return {"analysis": full_analysis_result}


# --- CORRECTED CHAT ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Provides a conversational chat interface for general security questions.
    It uses the chat_with_security_assistant tool to answer.
    """
    print(f"--- Received chat query: '{request.question[:75]}...' ---")
    
    try:
        # Call the dedicated chat assistant function with the user's question and history
        answer = chat_with_security_assistant(
            question=request.question,
            chat_history=request.chat_history
        )
        
        # Return the AI's answer in a JSON response
        return {"answer": answer}

    except Exception as e:
        # If anything goes wrong during the LLM call, return a server error
        print(f"ERROR in /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error during chat processing: {str(e)}")