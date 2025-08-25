import logging
import sys
import os
from fastapi import FastAPI

# Configure logging at the very beginning
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from pydantic import BaseModel

# Add the root directory of the Bot project to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.intelligent_orchestrator import orchestrate_chat

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    agent_response = orchestrate_chat(request.message, request.session_id)
    return {"response": agent_response}

from fastapi import Request, Response
from utils.whatsapp_handler import whatsapp_handler
import os

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