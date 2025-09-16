import os
import re
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional

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
import json

class TravelInfo(BaseModel):
    """Pydantic model for travel insurance information."""
    destination: Optional[str] = Field(None, description="The travel destination.")
    travel_duration_days: Optional[int] = Field(None, description="The duration of the travel in days (must be between 1 and 364).")
    pre_existing_medical_conditions: Optional[str] = Field(None, description="Answer to whether the user has pre-existing medical conditions ('yes' or 'no').")
    budget_preference: Optional[str] = Field(None, description="User's budget preference ('budget-friendly' or 'comprehensive').")
    is_ambiguous: Optional[bool] = Field(None, description="Set to true if the user's answer is ambiguous and a clear value cannot be extracted.")

    @field_validator('travel_duration_days')
    def validate_duration(cls, v):
        if v is not None and not (1 <= v <= 364):
            raise ValueError('Travel duration must be between 1 and 364 days.')
        return v
    
    @field_validator('pre_existing_medical_conditions')
    def validate_medical_conditions(cls, v):
        if v is not None and v.lower() not in ['yes', 'no']:
            raise ValueError("Pre-existing medical conditions must be 'yes' or 'no'.")
        return v

    @field_validator('budget_preference')
    def validate_budget(cls, v):
        if v is not None and v.lower() not in ['budget-friendly', 'comprehensive']:
            raise ValueError("Budget preference must be 'budget-friendly' or 'comprehensive'.")
        return v

def run_travel_agent(user_message: str, chat_history: list, session_id: str):
    """
    Handles the conversation for travel insurance inquiries using an LLM.
    """
    session = get_session(session_id)
    # Get only the travel-specific info, not the whole collected_info object
    travel_info = session.get("collected_info", {}).get("travel_info", {})
    
    # Define the required information
    required_info = ['destination', 'travel_duration_days', 'pre_existing_medical_conditions', 'budget_preference']

    chain = llm.with_structured_output(TravelInfo, method="function_calling")
    
    prompt = [
        SystemMessage(
            content=f"""You are a friendly and helpful travel insurance assistant. Your goal is to collect the following information from the user: {', '.join(required_info)}.

- For `travel_duration_days`, you must get a number between 1 and 364.
- For `pre_existing_medical_conditions`, you must get a 'yes' or 'no'.
- For `budget_preference`, you must get 'budget-friendly' or 'comprehensive'.

CRITICAL INTERPRETATION: You must interpret common colloquialisms. For example, 'nope', 'nah', 'not really' should be extracted as 'no'. 'yep', 'sure', 'i do' should be extracted as 'yes'.

AMBIGUITY HANDLING: Prioritize extracting a value if one is present, even if the user expresses uncertainty (e.g., for 'maybe 600 days', extract 600). Only set the `is_ambiguous` flag to `true` if the user provides a pure non-answer with no extractable data (e.g., 'I don't know', 'not sure').

Ask one question at a time. Be conversational. Once all information is collected, confirm with the user and tell them you will now find a recommendation.

Current collected information: {travel_info}
"""
        ),
        HumanMessage(content=user_message),
    ]
    
    # Call the LLM
    try:
        result = chain.invoke(prompt)
        
        # Handle cases where the user is unsure
        if result.is_ambiguous:
            missing_info = [key for key in required_info if key not in travel_info]
            first_missing = missing_info[0] if missing_info else None
            
            if first_missing == 'pre_existing_medical_conditions':
                return {
                    "output": "I understand you might be unsure. To help find the right plan, please answer with just 'yes' or 'no'. This is important for determining your coverage options.",
                    "stage": "travel_inquiry"
                }
            elif first_missing == 'budget_preference':
                return {
                    "output": "No problem. To recommend the best plan, could you let me know if you'd generally prefer a 'budget-friendly' option with essential coverage, or a more 'comprehensive' plan with maximum benefits?",
                    "stage": "travel_inquiry"
                }
            else: # Fallback for other ambiguous answers
                return {
                    "output": "I see you're not sure. Could you please try to provide the information so I can find the best plan for you?",
                    "stage": "travel_inquiry"
                }

        # Update collected info with new non-null values
        for key, value in result.model_dump().items():
            if value is not None and key != 'is_ambiguous':
                travel_info[key] = value
                
    except ValidationError as e:
        # Extract the first validation error message to be user-friendly.
        try:
            error_message = e.errors()[0]['msg']
            # The message from our validator is already user-friendly.
            user_facing_error = error_message.replace("Value error, ", "")
        except (IndexError, KeyError):
            user_facing_error = "The information provided was not in the correct format."
        
        logger.error(f"Pydantic validation error in travel agent: {user_facing_error}")
        # Return the specific validation error to the user.
        return {
            "output": f"It looks like there was an issue with your answer. {user_facing_error} Could you please provide it again?",
            "stage": "travel_inquiry"
        }

    except Exception as e:
        logger.error(f"Error in travel agent LLM call: {e}")
        # Return a friendly error message for other types of errors.
        return {
            "output": "I'm having a little trouble understanding that. Could you please rephrase?",
            "stage": "travel_inquiry"
        }

    # Persist the updated info
    set_collected_info(session_id, "travel_info", travel_info)
    
    logger.info(f"Travel info collected for {session_id}: {travel_info}")
    
    # Check if all required information is collected
    missing_info = [key for key in required_info if key not in travel_info]
    
    logger.info(f"All collected: {not missing_info}")
    
    # If all info is collected, move to the recommendation stage
    if not missing_info:
        from app.session_manager import set_stage
        set_stage(session_id, "recommendation")
        logger.info(f"Moving to recommendation stage for session {session_id}")
        
        # Call the recommendation agent
        from .recommendation_agent import get_recommendation
        recommendation = get_recommendation(session_id, "TRAVEL")
        
        # Store the recommended plan in the session context for future reference
        from app.session_manager import update_conversation_context
        if recommendation and recommendation.get('plan') != 'not available':
            update_conversation_context(session_id, recommended_plan=recommendation.get('plan'))

        # Generate the recommendation message
        from .rec_retriever_agent import get_recommendation_message
        try:
            plan_tier = recommendation['plan']
            recommendation_message = get_recommendation_message("TRAVEL", plan_tier)
            
            # Add the personalized reason from the recommendation
            if 'recommendation_reason' in recommendation and 'next_step_prompt' in recommendation:
                personalized_response = (
                    f"Based on your preferences, I recommend the **{plan_tier}** plan.\n\n"
                    f"{recommendation['recommendation_reason']}\n\n"
                    f"Here's a quick look at the benefits:\n{recommendation_message}\n\n"
                    f"{recommendation['next_step_prompt']}"
                )
                return {
                    "output": personalized_response,
                    "stage": "recommendation"
                }
            else:
                return {
                    "output": f"Based on your needs, I recommend the **{plan_tier}** plan. Here are the benefits:\n{recommendation_message}",
                    "stage": "recommendation"
                }

        except Exception as e:
            logger.error(f"Error generating recommendation message: {e}")
            return {
                "output": "I've collected all your details, but I'm having trouble generating a recommendation right now. Please try again shortly.",
                "stage": "recommendation"
            }

    # If info is missing, ask the next question
    else:
        next_question = {
            'destination': "Great! Where will you be traveling to? 🌍",
            'travel_duration_days': "How many days will you be traveling for?",
            'pre_existing_medical_conditions': "Do you have any pre-existing medical conditions? (yes/no)",
            'budget_preference': "Are you looking for a budget-friendly plan or a more comprehensive one?"
        }
        
        # Find the first missing item and ask the corresponding question
        question_to_ask = next_question.get(missing_info[0], "I need a little more information, but I'm not sure what to ask. Could you tell me more about your trip?")
        
        return {
            "output": question_to_ask,
            "stage": "travel_inquiry"
        }
