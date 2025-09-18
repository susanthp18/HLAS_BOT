import logging
import sys
import os
from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException
from utils.zoom.engagement import EngagementManager
from utils.whatsapp_handler import whatsapp_handler
import os
from typing import Dict
from functools import partial
# Configure logging at the very beginning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from pydantic import BaseModel

# Add the root directory of the Bot project to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.intelligent_orchestrator import orchestrate_chat

app = FastAPI()

# --- API Models ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class MessageRequest(BaseModel):
    text: str

class SessionResponse(BaseModel):
    status: str
    message: str
    session_id: str

class InitiateEngagement(BaseModel):
    nick_name: str = "Guest User"
    email: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    agent_response = orchestrate_chat(request.message, request.session_id)
    return {"response": agent_response}

@app.get("/meta-whatsapp")
def meta_whatsapp_webhook_verification(request: Request):
    """
    Enhanced webhook verification with comprehensive error handling.
    """
    return whatsapp_handler.verify_webhook(request)

@app.post("/meta-whatsapp")
async def meta_whatsapp_webhook(request: Request):
    """
    Enhanced webhook handler for incoming WhatsApp messages with production-grade features.
    """
    return await whatsapp_handler.process_webhook(request)

@app.get("/whatsapp/health")
def whatsapp_health_check():
    """
    WhatsApp-specific health check with detailed status information.
    """
    return whatsapp_handler.get_health_status()

# --- SESSION STORE (In-Memory Dictionary Approach) ---
# This dictionary holds the state of all active conversations.
# Key: A unique session_id (e.g., a customer's phone number).
# Value: The EngagementManager instance handling that session's state and connections.
#
# !! PRODUCTION NOTES !!
# This is great for development. However, in production, if you run your app
# with multiple workers (e.g., gunicorn -w 4), each worker will have its own
# separate dictionary, and your app will not work correctly. For production,
# this should be replaced with an external store like Redis.

# --- LOCAL TEST MESSAGING FLOW ---
# current flow will be triggered through API endpoint to initiate engagement with zoom agent
# The EngagementManager will handle the connection and messaging.
# Messages to the zoom agent will have to be sent through the API endpoint.
# The responses from the agent however can be received to the the Whatsapp number of the user with which engagement is initiated.

active_engagements: Dict[str, EngagementManager] = {}


# --- Configuration ---
BASE_URL = "https://us01cciapi.zoom.us"

# --- Callback & Cleanup Logic ---
async def handle_agent_response(session_id: str, message: dict | str):
    """Callback triggered by EngagementManager for agent messages."""
    logger.info(f"Callback for session '{session_id}': Received message from agent: {message}")
    # --- FORWARD MESSAGE TO CUSTOMER THROUGH WHATSAPP ---
    whatsapp_handler._send_message(session_id, message)
    
    event = message.get("event") if isinstance(message, dict) else None
    if event == "consumer_disconnected":
        logger.info(f"Agent ended chat for session '{session_id}'. Cleaning up.")
        await close_engagement_and_cleanup(session_id)


async def close_engagement_and_cleanup(session_id: str):
    """Helper to gracefully close and remove an engagement from the session store."""
    manager = active_engagements.get(session_id)
    if manager:
        await manager.close()
        del active_engagements[session_id]
        logger.info(f"Successfully closed and cleaned up session: {session_id}")

# --- API Endpoints ---
@app.post("/chats/{session_id}/start", response_model=SessionResponse)
async def start_chat_session(session_id: str, initiate_engagement: InitiateEngagement, background_tasks: BackgroundTasks):
    """
    1. Creates a new EngagementManager instance.
    2. Stores it in our `active_engagements` session store.
    3. Starts the connection process in the background.
    """
    if session_id in active_engagements:
        raise HTTPException(status_code=409, detail=f"Chat session '{session_id}' is already active.")

    logger.info(f"Creating and storing new engagement for session: {session_id}")
    callback_with_session_context = partial(handle_agent_response, session_id)
    manager = EngagementManager(
        nick_name=initiate_engagement.nick_name,
        email=initiate_engagement.email,
        base_api_url=BASE_URL,
        on_agent_message_callback=callback_with_session_context
    )
    # Add the new manager to our session store
    active_engagements[session_id] = manager
    background_tasks.add_task(manager.initiate_engagement)
    return {"status": "success", "message": "Engagement initiation started.", "session_id": session_id}


@app.post("/chats/{session_id}/message", response_model=SessionResponse)
async def send_customer_message(session_id: str, message: MessageRequest):
    """
    1. Looks up the EngagementManager from the session store using session_id.
    2. Calls the send_message method on that specific instance. This send the customer's message to the agent.
    """
    # Retrieve the specific manager instance from our session store
    manager = active_engagements.get(session_id)
    if not manager:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found or is inactive.")

    if not manager.is_agent_connected:
        raise HTTPException(status_code=400, detail="Agent has not connected yet.")

    await manager.send_message(message.text)
    return {"status": "success", "message": "Message sent.", "session_id": session_id}


@app.post("/chats/{session_id}/end", response_model=SessionResponse)
async def end_chat_session(session_id: str):
    """
    1. Looks up the EngagementManager from the session store.
    2. Calls the cleanup function to close connections and remove it.
    """
    if session_id not in active_engagements:
        raise HTTPException(status_code=404, detail=f"Chat session '{session_id}' not found.")
    await close_engagement_and_cleanup(session_id)
    return {"status": "success", "message": "Chat session has been ended.", "session_id": session_id}
