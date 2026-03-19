import json
import logging
import sqlite3
from datetime import datetime, timezone

# Logger :- 
logger = logging.getLogger(__name__)

# Constants :- 
DB_PATH = "data/query_logs.db"
DEFAULT_WORKSPACES: list[str] = ["default", "got", "dexter"]

# Main function

def _get_connection() -> sqlite3.Connection:
    """
    Returns a new database connection.
    """
    conn = sqlite3.connect(DB_PATH)
    return conn

# DATABASE INITIALISATION

def _initialize_db() -> None:
    conn = _get_connection()
    cursor = conn.cursor()

    # Original query logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT    NOT NULL,
            query          TEXT    NOT NULL,
            workspace      TEXT,
            latency_ms     REAL,
            results_count  INTEGER,
            docs           TEXT
        )
    """)

    # Conversation memory table
    # Each row is one turn (one user message + one assistant reply)
    # conversation_id groups turns belonging to the same conversation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id   TEXT    NOT NULL,
            workspace         TEXT    NOT NULL,
            user_message      TEXT    NOT NULL,
            assistant_message TEXT    NOT NULL,
            created_at        TEXT    NOT NULL
        )
    """)
    # conversation_id — groups all turns of one chat session together
    
    # workspaces
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            description TEXT,
            created_at  TEXT    NOT NULL
        )
    """)

    # document sources
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_sources (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            filename   TEXT    NOT NULL,
            workspace  TEXT    NOT NULL,
            indexed_at TEXT    NOT NULL,
            UNIQUE(filename, workspace)
        )
    """)

    for workspace_name in DEFAULT_WORKSPACES:
        cursor.execute("""
            INSERT OR IGNORE INTO workspaces (name, description, created_at)
            VALUES (?, ?, ?)
        """, (
            workspace_name,
            # name — the workspace identifier string
 
            f"Default workspace: {workspace_name}",
            # description — auto-generated description for seed workspaces
 
            datetime.now(timezone.utc).isoformat(),
            # created_at — current UTC time as ISO 8601 string
        ))
    conn.commit()
    conn.close()
    logger.info("Database initialised — all tables ready.")


# WORKSPACE OPERATIONS (Phase D)
def create_workspace(name: str, description: str = "") -> bool:
    """
    Inserts a new workspace into the workspaces table.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
 
        cursor.execute("""
            INSERT INTO workspaces (name, description, created_at)
            VALUES (?, ?, ?)
        """, (
            name,
            description,
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        conn.close()
 
        logger.info("Workspace created | name='%s'", name)
        return True
 
    except sqlite3.IntegrityError:
        logger.warning("Workspace already exists | name='%s'", name)
        conn.close()
        return False
 
    except Exception as e:
        logger.error("Failed to create workspace '%s': %s", name, e)
        conn.close()
        return False

def get_all_workspaces() -> list[dict]:
    """
    Returns all workspaces from the registry as a list of dicts.
 
    This is what GET /workspaces calls — returns the LIVE list
    from SQLite, not a hardcoded constant.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
 
        cursor.execute("""
            SELECT name, description, created_at
            FROM workspaces
            ORDER BY created_at ASC
        """)
        # ORDER BY created_at ASC — oldest workspaces first.
        # This means default workspaces always appear at the top of the list.
 
        rows = cursor.fetchall()
        # fetchall() retrieves ALL matching rows as a list of tuples. 
        conn.close()
 
        return [
            {
                "name":        row[0],
                "description": row[1],
                "created_at":  row[2],
            }
            for row in rows
        ]
        # List comprehension — converts each tuple into a dict.
        # row[0] = name, row[1] = description, row[2] = created_at
 
    except Exception as e:
        logger.error("Failed to fetch workspaces: %s", e)
        return []

def workspace_exists(name: str) -> bool:
    """
    Checks whether a workspace with the given name exists in the registry.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
 
        cursor.execute("""
            SELECT 1 FROM workspaces WHERE name = ?
        """, (name,))
        row = cursor.fetchone()
        # fetchone() returns the first matching row, or None if no match.
 
        conn.close()
 
        return row is not None
        # row is not None means a matching row was found → workspace exists.
        # row is None means no match → workspace does not exist.
 
    except Exception as e:
        logger.error("Failed to check workspace existence '%s': %s", name, e)
        return False


def delete_workspace(name: str) -> bool:
    """
    Deletes a workspace from the registry.
    Default workspaces ("default", "got", "dexter") cannot be deleted.
    """
    if name in DEFAULT_WORKSPACES:
        # Protect default workspaces from deletion.
        # Deleting "default" would break any client that doesn't specify
        logger.warning(
            "Attempted to delete protected default workspace '%s'", name
        )
        return False
 
    try:
        conn = _get_connection()
        cursor = conn.cursor()
 
        cursor.execute("""
            DELETE FROM workspaces WHERE name = ?
        """, (name,))
        # DELETE FROM workspaces WHERE name = ?
        # Removes the row matching this workspace name.
 
        deleted: bool = cursor.rowcount > 0
        # cursor.rowcount — number of rows affected by the last operation.
        # If > 0: at least one row was deleted → success.
        # If == 0: no row matched this name → workspace didn't exist.
 
        conn.commit()
        conn.close()
 
        if deleted:
            logger.info("Workspace deleted from registry | name='%s'", name)
        else:
            logger.warning(
                "Delete attempted but workspace not found | name='%s'", name
            )
 
        return deleted
 
    except Exception as e:
        logger.error("Failed to delete workspace '%s': %s", name, e)
        return False


def register_document_source(filename: str, workspace: str) -> None:
    """
    Records that a file has been indexed into a workspace.
 
    Called by document_processor.py after successfully indexing a document.
    This is what makes the workspace-to-file mapping persistent and correct.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
 
        cursor.execute("""
            INSERT OR IGNORE INTO document_sources (filename, workspace, indexed_at)
            VALUES (?, ?, ?)
        """, (
            filename,
            workspace,
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
        conn.close()
 
        logger.info(
            "Document source registered | filename='%s' | workspace='%s'",
            filename,
            workspace,
        )
 
    except Exception as e:
        logger.error(
            "Failed to register document source '%s' in workspace '%s': %s",
            filename,
            workspace,
            e,
        )

def get_workspace_files_from_db(workspace: str) -> list[str]:
    """
    Returns all filenames registered under the given workspace.
 
    Called by keyword_search.py to find which files to include
    in the BM25 index for this workspace.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
 
        cursor.execute("""
            SELECT filename FROM document_sources
            WHERE workspace = ?
            ORDER BY indexed_at ASC
        """, (workspace,))
        # SELECT filename — we only need the filename strings, not the full row.
 
        rows = cursor.fetchall()
        conn.close()
 
        filenames: list[str] = [row[0] for row in rows]
        # Each row is a tuple: (filename,)
        # row[0] extracts just the filename string.
 
        logger.debug(
            "Found %d files for workspace '%s': %s",
            len(filenames),
            workspace,
            filenames,
        )
 
        return filenames
 
    except Exception as e:
        logger.error(
            "Failed to fetch files for workspace '%s': %s", workspace, e
        )
        return []

def get_workspace_names() -> list[str]:
    """
    Returns just the workspace names as a plain list of strings.
 
    Used by:
        - query_planner.py — to find all non-default workspaces
          for cross-workspace comparison queries
        - main.py GET /workspaces — for the simple workspaces list response
    """
    workspaces = get_all_workspaces()
    return [w["name"] for w in workspaces]

# QUERY LOG OPERATIONS

def log_query(
    query: str,
    docs: list,
    latency: float,
    workspace: str = "default",
) -> None:
    """
    Persists a search query and its results to the query_logs table.
 
    Called by search.py after every completed search pipeline run.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO query_logs (timestamp, query, workspace, latency_ms, results_count, docs)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            query,
            workspace,
            round(latency, 2),
            len(docs),
            json.dumps(docs, default=str)
        ))

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error("Failed to log query: %s", e)


def get_recent_logs(limit: int = 20):
    """
    Returns the most recent query log entries.
 
    Called by main.py GET /logs endpoint.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, query, workspace, latency_ms, results_count
            FROM query_logs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "timestamp": row[0],
                "query": row[1],
                "workspace": row[2],
                "latency_ms": row[3],
                "results_count": row[4]
            }
            for row in rows
        ]

    except Exception as e:
        logger.error("Failed to fetch logs: %s", e)
        return []

# CONVERSATION OPERATIONS

def save_conversation_turn(
    conversation_id: str,
    workspace: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """
    Saves one completed conversation turn (user + assistant) to SQLite.
 
    Called by answer_generator.py after the LLM finishes streaming.
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO conversations (conversation_id, workspace, user_message, assistant_message, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            conversation_id,
            workspace,
            user_message,
            assistant_message,
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        conn.close()
        
        logger.debug(
            "Conversation turn saved | conversation_id = '%s'",
            conversation_id,
        )

    except Exception as e:
        logger.error(
            "Failed to save conversation turn: %s",
            e,
        )


def get_conversation_history(
    conversation_id: str,
    limit: int = 6,
) -> list[dict]:
    # Load the last N turns of a conversation
    # limit=6 means last 6 turns (6 user + 6 assistant messages)
    # keeping history short prevents the LLM prompt from getting too long
    try:
        conn = _get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_message, assistant_message
            FROM conversations
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (conversation_id, limit))

        rows = cursor.fetchall()
        conn.close()

        # reverse so oldest turn comes first — correct chronological order
        # The LLM needs to read the conversation in the order it happened.
        rows.reverse()

        return [
            {"user": row[0], "assistant": row[1]}
            for row in rows
        ]

    except Exception as e:
        logger.error(
            "Failed to fetch conversation history: %s",
            e,
        )
        return []


# Initialize both tables when this module is first imported
_initialize_db()