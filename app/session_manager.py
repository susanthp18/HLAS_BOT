from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from app.config import MAX_CONTEXT_MESSAGES

logger = logging.getLogger(__name__)

# Enhanced session storage with metadata
SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Dict[str, Any]:
    """
    Retrieves a session or creates a new one if it doesn't exist with enhanced metadata.
    """
    if session_id not in SESSIONS:
        now = datetime.now()
        SESSIONS[session_id] = {
            "chat_history": [],
            "stage": "initial",
            "created_at": now,
            "last_active": now,
            "message_count": 0,
            "user_preferences": {},
            "collected_info": {},
            "conversation_context": {
                "current_agent": None,
                "primary_product": None,
                "has_greeted": False,
                "information_collected": False,
                "last_intent": None,
                "error_count": 0
            }
        }
        logger.info(f"Created new session: {session_id}")
    else:
        # Update last active time
        SESSIONS[session_id]["last_active"] = datetime.now()
    
    return SESSIONS[session_id]

def update_session(session_id: str, user_message: str, agent_response: str):
    """
    Updates the chat history and session metadata for a given session.
    """
    session = get_session(session_id)
    chat_history = session["chat_history"]
    
    # Add messages to history
    chat_history.append({"role": "user", "content": user_message, "timestamp": datetime.now()})
    chat_history.append({"role": "assistant", "content": agent_response, "timestamp": datetime.now()})
    
    # Update session metadata
    session["message_count"] += 1
    session["last_active"] = datetime.now()
    
    # Trim history to keep it within the configured limit
    if len(chat_history) > MAX_CONTEXT_MESSAGES * 2:
        # Keep the last MAX_CONTEXT_MESSAGES pairs of messages
        session["chat_history"] = chat_history[-(MAX_CONTEXT_MESSAGES * 2):]
        logger.debug(f"Trimmed chat history for session {session_id}")
    
    logger.debug(f"Updated session {session_id}: {session['message_count']} total messages")

def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Returns the chat history for a given session.
    """
    session = get_session(session_id)
    return session["chat_history"]

def get_stage(session_id: str) -> str:
    """
    Returns the current stage for a given session.
    """
    session = get_session(session_id)
    return session["stage"]

def set_stage(session_id: str, stage: str):
    """
    Sets the stage for a given session.
    """
    session = get_session(session_id)
    old_stage = session.get("stage", "initial")
    session["stage"] = stage
    logger.info(f"Session {session_id} stage changed: {old_stage} -> {stage}")

def update_conversation_context(session_id: str, **kwargs):
    """
    Updates conversation context for better intelligence.
    """
    session = get_session(session_id)
    context = session["conversation_context"]
    
    for key, value in kwargs.items():
        context[key] = value  # Always update/add the key-value pair
    
    logger.debug(f"Updated conversation context for session {session_id}: {kwargs}")

def set_collected_info(session_id: str, info_type: str, value: Any):
    """
    Stores collected information for the session.
    """
    session = get_session(session_id)
    session["collected_info"][info_type] = value
    logger.debug(f"Stored {info_type} for session {session_id}")

def get_collected_info(session_id: str, info_type: str = None) -> Any:
    """
    Retrieves collected information from the session.
    """
    session = get_session(session_id)
    if info_type:
        return session["collected_info"].get(info_type)
    return session["collected_info"]

def increment_error_count(session_id: str):
    """
    Increments error count for session monitoring.
    """
    session = get_session(session_id)
    session["conversation_context"]["error_count"] += 1
    error_count = session["conversation_context"]["error_count"]
    
    if error_count > 5:
        logger.warning(f"High error count ({error_count}) for session {session_id}")
    
    return error_count

def cleanup_old_sessions(max_age_hours: int = 24):
    """
    Removes sessions older than specified hours.
    """
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    to_remove = []
    
    for session_id, session in SESSIONS.items():
        if session.get("last_active", datetime.now()) < cutoff_time:
            to_remove.append(session_id)
    
    for session_id in to_remove:
        del SESSIONS[session_id]
        logger.info(f"Cleaned up old session: {session_id}")
    
    return len(to_remove)

def get_session_stats() -> Dict[str, Any]:
    """
    Returns statistics about active sessions.
    """
    total_sessions = len(SESSIONS)
    active_sessions = 0
    total_messages = 0
    
    for session in SESSIONS.values():
        if session.get("last_active", datetime.min) > datetime.now() - timedelta(hours=1):
            active_sessions += 1
        total_messages += session.get("message_count", 0)
    
    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "total_messages": total_messages,
        "average_messages_per_session": total_messages / max(total_sessions, 1)
    }