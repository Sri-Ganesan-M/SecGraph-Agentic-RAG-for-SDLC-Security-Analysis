import sqlite3
import json
from typing import Dict, Optional

DB_FILE = "mcp_context.db"

def init_db():
    """Initializes the database and creates the context table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_context (
            file_path TEXT PRIMARY KEY,
            context_json TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("MCP Database initialized.")

def get_context(file_path: str) -> Optional[Dict]:
    """Retrieves the stored context for a given file path."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    res = cursor.execute("SELECT context_json FROM file_context WHERE file_path = ?", (file_path,)).fetchone()
    conn.close()
    return json.loads(res[0]) if res else None

def save_context(file_path: str, context_obj: Dict):
    """Saves or updates the context for a given file path."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO file_context (file_path, context_json) VALUES (?, ?)",
        (file_path, json.dumps(context_obj))
    )
    conn.commit()
    conn.close()