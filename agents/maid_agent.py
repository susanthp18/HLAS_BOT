import os
import re
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from typing import List, Type, Optional

from app.config import llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from app.session_manager import set_collected_info, get_collected_info

logger = logging.getLogger(__name__)



from app.session_manager import get_session, update_session
from utils.llm_services import llm
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import json

class MaidInfo(BaseModel):
    """Pydantic model for maid insurance information."""
    contract_duration: Optional[int] = Field(None, description="The duration of the maid's contract in months.")
    personal_accident_coverage: Optional[int] = Field(None, description="The desired personal accident coverage amount.")
    response: str = Field(..., description="The response to the user.")

def run_maid_agent(user_message: str, chat_history: list, session_id: str):
    """
    Handles the conversation for maid insurance inquiries using an LLM.
    """
    session = get_session(session_id)
    collected_info = session.get("collected_info", {})
    
    # Information to be collected
    required_info = ["contract_duration", "personal_accident_coverage"]
    
    # Use an LLM to have a conversation and collect the information
    chain = llm.with_structured_output(MaidInfo, method="function_calling")
    
    prompt = [
        SystemMessage(
            content=f"""You are a friendly and helpful maid insurance assistant. Your goal is to collect the following information from the user: {', '.join(required_info)}.

Current collected information: {collected_info}
Conversation history: {chat_history}

IMPORTANT EXTRACTION RULES:
- If user just says a number (like "12", "24", "36") when you asked about contract duration, extract it as contract_duration (in months)
- If user mentions coverage amounts or insurance amounts, extract as personal_accident_coverage
- Always acknowledge what they provided and ask for the next missing piece

Ask for the next piece of information in a conversational way. If the user provides information you were not asking for, acknowledge it and continue with the required information.
"""
        ),
        HumanMessage(content=user_message),
    ]
    
    response = chain.invoke(prompt)
    
    # Update the collected information
    for key, value in response.model_dump().items():
        if key != "response" and value:
            collected_info[key] = value
    
    # Store collected info in session using the proper method
    set_collected_info(session_id, "maid_info", collected_info)
    
    # Debug logging
    logger.info(f"Maid info collected for {session_id}: {collected_info}")
    logger.info(f"Required info: {required_info}")
    logger.info(f"All collected: {all(key in collected_info for key in required_info)}")
    
    # Check if all information has been collected NOW (after processing current message)
    if all(key in collected_info for key in required_info):
        from app.session_manager import set_stage
        set_stage(session_id, "recommendation")
        logger.info(f"Moving to recommendation stage for session {session_id}")
        
        # Automatically move to recommendation flow
        try:
            from .recommendation_agent import get_recommendation
            from .rec_retriever_agent import get_recommendation_message
            
            # Get AI recommendation
            recommendation = get_recommendation(session_id, "MAID")
            
            plan_tier = recommendation.get("plan", "Standard")
            
            # Store the recommended plan in conversation context
            from app.session_manager import update_conversation_context
            update_conversation_context(session_id, recommended_plan=plan_tier)
            
            # Get comprehensive recommendation message with benefits
            recommendation_message = get_recommendation_message("MAID", plan_tier)
            
            # Add purchase guidance to the recommendation
            recommendation_message += "\n\n" + """**What's Next?**
🔍 Ask me about **different plan options** (Basic, Enhanced, Premier, Exclusive)
❓ Ask **any questions** about coverage details  
💳 Say **"I want to proceed with purchase"** when you're ready

What would you like to do?"""
            
            return recommendation_message
        except Exception as e:
            logger.error(f"Error in recommendation flow: {str(e)}")
            # Return error message instead of hardcoded fallback
            return """I'm having trouble generating a recommendation right now. 😅 

This could be due to:
🔧 A temporary system issue
📡 Connection problems with our recommendation service

Please try again in a moment, or feel free to ask me any specific questions about our maid insurance plans! I can help you with:

🏠 **Plan comparisons** (Basic, Enhanced, Premier, Exclusive)
💰 **Pricing information** 
📋 **Coverage details**
❓ **Any specific questions** about benefits

What would you like to know more about?"""
    
    return response.response
