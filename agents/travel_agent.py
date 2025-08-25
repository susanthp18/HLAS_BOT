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

class TravelInfo(BaseModel):
    """Pydantic model for travel information."""
    destination: Optional[str] = Field(None, description="The user's travel destination.")
    start_date: Optional[str] = Field(None, description="The start date of the trip.")
    end_date: Optional[str] = Field(None, description="The end date of the trip.")
    party_size: Optional[int] = Field(None, description="The number of people traveling.")
    response: str = Field(..., description="The response to the user.")

def run_travel_agent(user_message: str, chat_history: list, session_id: str):
    """
    Handles the conversation for travel insurance inquiries using an LLM.
    """
    session = get_session(session_id)
    collected_info = session.get("collected_info", {})
    
    # Information to be collected
    required_info = ["destination", "start_date", "end_date", "party_size"]
    
    # Use an LLM to have a conversation and collect the information
    chain = llm.with_structured_output(TravelInfo, method="function_calling")
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    prompt = [
        SystemMessage(
            content=f"""You are a friendly and helpful travel insurance assistant. Your goal is to collect the following information from the user: {', '.join(required_info)}.

Today's date is {today_date}. All travel dates must be in the future.

Current collected information: {collected_info}
Conversation history: {chat_history}

IMPORTANT EXTRACTION RULES:
- If user just says a number (like "2", "3", "4") when you asked about party size, extract it as party_size
- If user provides dates, extract them as start_date and end_date. Validate that these dates are after today's date ({today_date}). If a date is in the past, inform the user and ask for a valid future date.
- If user mentions a place/country, extract it as destination
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
    set_collected_info(session_id, "travel_info", collected_info)
    
    # Debug logging
    logger.info(f"Travel info collected for {session_id}: {collected_info}")
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
            recommendation = get_recommendation(session_id, "TRAVEL")
            
            plan_tier = recommendation.get("plan", "Standard")
            
            # Store the recommended plan in conversation context
            from app.session_manager import update_conversation_context
            update_conversation_context(session_id, recommended_plan=plan_tier)
            
            # Get comprehensive recommendation message with benefits
            recommendation_message = get_recommendation_message("TRAVEL", plan_tier)
            
            # Add purchase guidance to the recommendation
            recommendation_message += "\n\n" + """**What's Next?**
🔍 Ask me about **different plan options** (Basic, Silver, Gold, Platinum)
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

Please try again in a moment, or feel free to ask me any specific questions about our travel insurance plans! I can help you with:

🌍 **Plan comparisons** (Silver, Gold, Platinum)
💰 **Pricing information** 
📋 **Coverage details**
❓ **Any specific questions** about benefits

What would you like to know more about?"""
    
    return response.response
