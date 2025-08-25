"""
Fallback and Error Recovery System for HLAS Insurance Chatbot
============================================================

This module provides comprehensive fallback mechanisms and error recovery
for production-grade WhatsApp chatbot conversations.
"""

import logging
import random
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.session_manager import increment_error_count, get_session, update_conversation_context

logger = logging.getLogger(__name__)

class FallbackManager:
    """
    Manages fallback responses and error recovery for the chatbot.
    """
    
    def __init__(self):
        self.fallback_responses = {
            "general_error": [
                "Oops! Something went wrong on my end. 😅 Could you please try again?",
                "I'm having a small technical hiccup. Please send your message again! 🔧",
                "Sorry about that! There was a brief issue. Could you resend your question? 💻"
            ],
            "input_validation_error": [
                "I didn't quite catch that! 😅 Could you please rephrase your question?",
                "Hmm, I'm not sure I understand. Could you tell me what you're looking for? 🤔",
                "Sorry, I couldn't understand that message. How can I help you today? 💬"
            ],
            "agent_error": [
                "I'm having trouble processing that right now. 😅 Could you try rephrasing your question?",
                "Let me try to help differently. What specific insurance information do you need? 🤝",
                "I want to make sure I give you the right help. Could you tell me more about what you're looking for? 📋"
            ],
            "timeout_error": [
                "Sorry for the delay! 🕐 I'm still here to help. What can I do for you?",
                "I'm back! 😊 How can I assist you with your insurance needs?",
                "Thanks for your patience! What insurance question can I help you with? 🌟"
            ],
            "too_many_errors": [
                "I'm having some technical difficulties today. 😔 For immediate assistance, please contact our support team at support@hlas.com.sg or call +65 6227 7888.",
                "I apologize for the repeated issues. 🙏 Our human agents can help you right away at support@hlas.com.sg or +65 6227 7888.",
                "Let me connect you with our support team for better assistance. 📞 Please contact support@hlas.com.sg or call +65 6227 7888."
            ],
            "off_topic": [
                "I specialize in helping with insurance questions! 😊 What can I help you with regarding Travel or Maid insurance?",
                "I'm here to assist with your insurance needs. How can I help with Travel or Maid insurance today? 🛡️",
                "Let's talk insurance! I can help you with Travel or Maid insurance. What would you like to know? ✈️🏠"
            ],
            "product_not_available": [
                "I currently specialize in Travel and Maid insurance! 🌟 Which of these would be helpful for you?",
                "Right now I can help with Travel and Maid insurance. 😊 Are either of these what you're looking for?",
                "I'm an expert in Travel and Maid insurance! ✈️🏠 Would you like to know more about either of these?"
            ]
        }
        
        self.escalation_triggers = {
            "high_error_count": 5,
            "repeated_failures": 3,
            "complex_queries": ["legal", "lawsuit", "complaint", "manager", "supervisor"]
        }
    
    def get_fallback_response(self, error_type: str, session_id: Optional[str] = None) -> str:
        """
        Get an appropriate fallback response based on error type and context.
        """
        try:
            # Check if we should escalate to human support
            if session_id and self.should_escalate(session_id, error_type):
                return self.get_escalation_response()
            
            # Get appropriate fallback response
            responses = self.fallback_responses.get(error_type, self.fallback_responses["general_error"])
            response = random.choice(responses)
            
            # Log the fallback usage
            logger.info(f"Fallback response used: {error_type} for session {session_id}")
            
            # Update error count if session provided
            if session_id:
                increment_error_count(session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in fallback system: {str(e)}")
            return "I'm here to help with your insurance needs! 😊 What can I assist you with today?"
    
    def should_escalate(self, session_id: str, error_type: str) -> bool:
        """
        Determine if conversation should be escalated to human support.
        """
        try:
            session = get_session(session_id)
            error_count = session["conversation_context"]["error_count"]
            
            # Check error count threshold
            if error_count >= self.escalation_triggers["high_error_count"]:
                logger.warning(f"Escalating session {session_id} due to high error count: {error_count}")
                return True
            
            # Check for repeated agent errors
            if error_type == "agent_error" and error_count >= self.escalation_triggers["repeated_failures"]:
                logger.warning(f"Escalating session {session_id} due to repeated agent failures")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in escalation check: {str(e)}")
            return False
    
    def get_escalation_response(self) -> str:
        """
        Get escalation response for human handoff.
        """
        escalation_responses = [
            "I'd like to connect you with our specialist team for better assistance. 👥 Please contact support@hlas.com.sg or call +65 6227 7888.",
            "Let me get you in touch with our expert team! 🌟 Please reach out to support@hlas.com.sg or call +65 6227 7888.",
            "Our human specialists can provide you with personalized help. 😊 Contact support@hlas.com.sg or call +65 6227 7888."
        ]
        
        return random.choice(escalation_responses)
    
    def handle_agent_failure(self, session_id: str, agent_type: str, error_message: str) -> str:
        """
        Handle specific agent failures with contextual responses.
        """
        try:
            logger.error(f"Agent failure in {agent_type} for session {session_id}: {error_message}")
            
            # Update conversation context
            update_conversation_context(session_id, last_error=error_message, last_error_time=datetime.now())
            
            # Agent-specific fallback responses
            agent_fallbacks = {
                "travel_agent": [
                    "I'm having trouble with travel insurance right now. 😅 Could you try asking about general coverage or contact our support team?",
                    "Let me help differently with travel insurance! ✈️ What specific travel coverage question do you have?",
                    "I want to ensure you get the right travel insurance info. 🌟 Could you rephrase your question?"
                ],
                "maid_agent": [
                    "I'm experiencing issues with maid insurance processing. 😅 Could you try rephrasing your question?",
                    "Let me help with maid insurance in a different way! 🏠 What specific coverage do you need to know about?",
                    "I want to give you accurate maid insurance information. 💙 Could you ask your question differently?"
                ],
                "payment_agent": [
                    "I'm having trouble with the payment system. 😅 Please try again or contact our support team for immediate assistance.",
                    "Let me help with payment processing! 💳 Could you provide your information again?",
                    "I want to ensure your payment goes through smoothly. 🛡️ Please try submitting your details once more."
                ]
            }
            
            fallback_messages = agent_fallbacks.get(agent_type, self.fallback_responses["agent_error"])
            return random.choice(fallback_messages)
            
        except Exception as e:
            logger.error(f"Error in agent failure handler: {str(e)}")
            return self.get_fallback_response("general_error", session_id)
    
    def detect_confusion_patterns(self, session_id: str, user_message: str) -> Optional[str]:
        """
        Uses LLM to detect patterns that indicate user confusion and provide helpful responses.
        """
        try:
            from app.config import llm
            from langchain_core.messages import SystemMessage, HumanMessage
            from pydantic import BaseModel, Field
            
            class ConfusionAnalysis(BaseModel):
                is_confused: bool = Field(description="Whether the user appears confused or needs help")
                confusion_type: str = Field(description="Type of confusion: what, how, help, unclear, repeat, or none")
                confidence: float = Field(description="Confidence score 0.0-1.0")
            
            # Get conversation context
            session = get_session(session_id)
            chat_history = session.get("chat_history", [])
            last_few_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
            
            confusion_chain = llm.with_structured_output(ConfusionAnalysis, method="function_calling")
            
            prompt = [
                SystemMessage(content="""You are an expert at detecting user confusion in insurance conversations.
                
Analyze if the user seems confused, lost, or needs help based on their message and recent conversation context.

Types of confusion to detect:
- "what": User asking what something is or means
- "how": User asking how something works or how to do something  
- "help": User explicitly asking for help or assistance
- "unclear": User saying they don't understand or are confused
- "repeat": User asking to repeat or clarify something
- "none": User is not confused, just having normal conversation

IMPORTANT CONTEXT RULES:
- If the system recently asked for corrections (e.g., "provide valid future date", "past date", "invalid format"), and the user provides new information, this is a CORRECTION, not confusion.
- Date corrections after validation errors are normal flow continuations.
- Short contextual responses like "2", "Japan", "yes", or date ranges are normal continuations, not confusion.
- Only flag as confused if the user genuinely seems lost or explicitly asks for clarification.
- If user was asked to fix something and provides different information, treat as correction attempt, not confusion."""),
                HumanMessage(content=f"Recent conversation: {last_few_messages}\n\nUser message: {user_message}")
            ]
            
            result = confusion_chain.invoke(prompt)
            
            if result.is_confused and result.confidence > 0.7:
                return self.get_confusion_response(result.confusion_type, session_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in LLM confusion detection: {str(e)}")
            return None
    
    def get_confusion_response(self, confusion_type: str, session_id: str) -> str:
        """
        Get response for detected confusion patterns.
        """
        confusion_responses = {
            "what": [
                "I'm here to help with Travel and Maid insurance! 😊 What specific information do you need?",
                "Let me explain! I can help with Travel insurance (for trips) and Maid insurance (for domestic helpers). What interests you? ✈️🏠",
                "I specialize in two types of insurance: Travel (for your trips) and Maid (for domestic helpers). Which would you like to know about? 🌟"
            ],
            "how": [
                "I'd be happy to guide you! 😊 Are you looking to learn about coverage, get a quote, or something else?",
                "Let me walk you through it! 👥 What specific process would you like help with?",
                "I can guide you step by step! 🌟 What would you like to know how to do?"
            ],
            "help": [
                "Absolutely! I'm here to help! 😊 I can assist with Travel and Maid insurance. What do you need?",
                "I'd love to help! 🌟 Tell me what insurance information you're looking for.",
                "Of course! I'm your insurance assistant. 💙 How can I help you today?"
            ],
            "confused": [
                "No worries! Let me make it clearer. 😊 I help with Travel insurance (for trips) and Maid insurance (for helpers). Which one?",
                "I understand! Let me simplify. 🌟 Do you need insurance for traveling or for your domestic helper?",
                "Let me explain better! 💙 I can help with trip insurance or helper insurance. Which interests you?"
            ],
            "repeat": [
                "Of course! 😊 I help with Travel insurance (for your trips) and Maid insurance (for domestic helpers). What can I tell you?",
                "Sure thing! 🌟 I specialize in Travel and Maid insurance. Which one would you like to know about?",
                "Happy to repeat! 💙 I assist with insurance for travel and domestic helpers. What specific info do you need?"
            ],
            "different": [
                "I focus on Travel and Maid insurance! 😊 While I don't handle other types, I'm expert in these two. Interested?",
                "Currently I specialize in Travel and Maid insurance. 🌟 These are my expertise areas. Would either help you?",
                "I'm specialized in Travel and Maid insurance! 💙 For other types, you'd need our general support team."
            ]
        }
        
        responses = confusion_responses.get(confusion_type, confusion_responses["help"])
        return random.choice(responses)

# Global fallback manager instance
fallback_manager = FallbackManager()

def get_fallback_response(error_type: str, session_id: Optional[str] = None) -> str:
    """Convenience function to get fallback response."""
    return fallback_manager.get_fallback_response(error_type, session_id)

def handle_agent_failure(session_id: str, agent_type: str, error_message: str) -> str:
    """Convenience function to handle agent failures."""
    return fallback_manager.handle_agent_failure(session_id, agent_type, error_message)

def detect_confusion(session_id: str, user_message: str) -> Optional[str]:
    """Convenience function to detect user confusion."""
    return fallback_manager.detect_confusion_patterns(session_id, user_message)

