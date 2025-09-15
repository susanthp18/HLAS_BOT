import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv

from app.config import MAX_CONTEXT_MESSAGES

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "hlas_bot"
COLLECTION_NAME = "conversations"

# --- Global MongoDB Client ---
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    conversations_collection = db[COLLECTION_NAME]
    # The ismaster command is cheap and does not require auth.
    client.admin.command('ismaster')
    logger.info("Successfully connected to MongoDB.")
    
    # Ensure a unique index on session_id to prevent duplicate sessions and ensure atomic upserts.
    conversations_collection.create_index("session_id", unique=True)
    logger.info("Ensured unique index on 'session_id' in conversations collection.")

except Exception as e:
    logger.critical(f"Failed to connect to MongoDB: {e}")
    client = None
    conversations_collection = None

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a session or creates a new one if it doesn't exist in MongoDB.
    """
    if conversations_collection is None:
        logger.error("MongoDB collection not available.")
        return None

    now = datetime.now()
    
    # Atomically find and update (or insert if not found)
    session = conversations_collection.find_one_and_update(
        {"session_id": session_id},
        {
            "$setOnInsert": {
                "chat_history": [],
                "stage": "initial",
                "created_at": now,
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
            },
            "$set": {
                "last_active": now
            }
        },
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    
    if session.get("message_count", 0) == 0:
        logger.info(f"Created new session: {session_id}")
        
    return session

def update_session(session_id: str, user_message: str, agent_response: str):
    """
    Updates the chat history and session metadata in MongoDB.
    """
    if conversations_collection is None: return

    now = datetime.now()
    history_updates = [
        {"role": "user", "content": user_message, "timestamp": now},
        {"role": "assistant", "content": agent_response, "timestamp": now}
    ]

    conversations_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"chat_history": {"$each": history_updates}},
            "$set": {"last_active": now},
            "$inc": {"message_count": 1}
        }
    )
    logger.debug(f"Updated session {session_id} in MongoDB.")


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Returns the most recent chat history for a given session from MongoDB.
    """
    if conversations_collection is None: return []

    # Use projection to get only the history, sliced to the last N pairs
    session = conversations_collection.find_one(
        {"session_id": session_id},
        {"chat_history": {"$slice": -(MAX_CONTEXT_MESSAGES * 2)}}
    )
    return session.get("chat_history", []) if session else []

def get_stage(session_id: str) -> str:
    """
    Returns the current stage for a given session from MongoDB.
    """
    session = get_session(session_id)
    return session.get("stage", "initial") if session else "initial"

def set_stage(session_id: str, stage: str):
    """
    Sets the stage for a given session in MongoDB.
    """
    if conversations_collection is None: return
    
    old_stage_doc = conversations_collection.find_one({"session_id": session_id}, {"stage": 1})
    old_stage = old_stage_doc.get("stage", "initial") if old_stage_doc else "initial"

    conversations_collection.update_one(
        {"session_id": session_id},
        {"$set": {"stage": stage, "last_active": datetime.now()}}
    )
    logger.info(f"Session {session_id} stage changed: {old_stage} -> {stage}")

def update_conversation_context(session_id: str, **kwargs):
    """
    Updates conversation context in MongoDB using dot notation.
    """
    if conversations_collection is None: return

    update_fields = {f"conversation_context.{key}": value for key, value in kwargs.items()}
    update_fields["last_active"] = datetime.now()

    conversations_collection.update_one(
        {"session_id": session_id},
        {"$set": update_fields}
    )
    logger.debug(f"Updated conversation context for session {session_id}: {kwargs}")


def set_collected_info(session_id: str, info_type: str, value: Any):
    """
    Stores collected information for the session in MongoDB.
    """
    if conversations_collection is None: return
    
    conversations_collection.update_one(
        {"session_id": session_id},
        {"$set": {f"collected_info.{info_type}": value, "last_active": datetime.now()}}
    )
    logger.debug(f"Stored {info_type} for session {session_id}")

def get_collected_info(session_id: str, info_type: str = None) -> Any:
    """
    Retrieves collected information from the session in MongoDB.
    """
    session = get_session(session_id)
    if not session: return {}

    collected_info = session.get("collected_info", {})
    if info_type:
        return collected_info.get(info_type)
    return collected_info

def increment_error_count(session_id: str) -> int:
    """
    Increments error count for a session in MongoDB and returns the new count.
    """
    if conversations_collection is None: return 0

    updated_session = conversations_collection.find_one_and_update(
        {"session_id": session_id},
        {
            "$inc": {"conversation_context.error_count": 1},
            "$set": {"last_active": datetime.now()}
        },
        return_document=ReturnDocument.AFTER
    )
    
    error_count = updated_session.get("conversation_context", {}).get("error_count", 0)
    
    if error_count > 5:
        logger.warning(f"High error count ({error_count}) for session {session_id}")
    
    return error_count

def cleanup_old_sessions(max_age_hours: int = 24):
    """
    Removes sessions from MongoDB older than specified hours.
    """
    if conversations_collection is None: return 0
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    result = conversations_collection.delete_many(
        {"last_active": {"$lt": cutoff_time}}
    )
    
    if result.deleted_count > 0:
        logger.info(f"Cleaned up {result.deleted_count} old session(s).")
        
    return result.deleted_count