"""
Robust LLM-Powered Conversation Flow Manager
==========================================

Uses intelligent conversation analysis to determine routing decisions
based on conversational flow and context, not just keyword matching.
"""

import logging
from typing import Dict, Any, Optional, List
from app.session_manager import get_session, update_conversation_context
from app.config import llm
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class ConversationFlowManager:
    """
    Intelligent conversation flow manager that understands conversational context
    and makes routing decisions based on the natural flow of conversation.
    """
    
    def __init__(self):
        self.decision_prompt = ChatPromptTemplate.from_template("""
You are an intelligent conversation flow analyzer for HLAS Insurance chatbot. Your task is to determine if the user is continuing the current conversation or switching to a new topic.

CONVERSATION CONTEXT:
- Current Agent: {current_agent}
- Last Agent Message: "{last_agent_message}"
- User's Response: "{user_message}"

CONVERSATION HISTORY (last 3 exchanges):
{conversation_history}

ANALYSIS GUIDELINES:
1.  **Direct Response**: If the user's message directly answers the last agent's question, the decision is **continue**.
2.  **Contextual Clues**: Short, contextual responses like "yes," "to the US," "next week," or "just two of us" are clear continuations.
3.  **Topic Switch**: A switch occurs only when the user explicitly introduces a new, unrelated topic (e.g., asking about a different type of insurance).
4.  **Initial Interaction**: If the `current_agent` is `None` or `UNKNOWN`, and the user provides a specific insurance type (e.g., "travel insurance"), the decision is **continue**, as the user is clarifying their initial request.
5.  **Ambiguity**: If the user's response is ambiguous, prefer to **continue** the current conversation rather than switching.

EXAMPLES:
- Agent: "Where are you traveling?" | User: "I'm going to Japan." → **continue**
- Agent: "How many people are traveling?" | User: "just me" → **continue**
- Agent: "Do you need travel insurance?" | User: "what about maid insurance?" → **switch**
- Agent: "I can help with travel or maid insurance." | User: "I need travel insurance." → **continue**

RESPONSE FORMAT (JSON):
{{
  "decision": "continue" | "switch" | "clarify",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<brief explanation>"
}}
""")

    def analyze_conversation_flow(
        self, 
        session_id: str, 
        user_message: str, 
        chat_history: List
    ) -> Dict[str, Any]:
        """
        Analyze conversation flow to determine if user is continuing current conversation
        or switching to a new topic.
        """
        try:
            # Get current context
            session = get_session(session_id)
            context = session.get("conversation_context", {})
            current_agent = context.get("primary_product", "None")
            
            # Extract last agent message and conversation history
            last_agent_message = self._extract_last_agent_message(chat_history)
            formatted_history = self._format_conversation_history(chat_history)
            
            # Only treat as new conversation if there's NO chat history at all
            if not chat_history:
                return {
                    "decision": "switch",  # Classify new conversation
                    "confidence": 1.0,
                    "reason": "First message - needs classification"
                }
            
            # If there's chat history but no specific agent yet, still analyze conversation flow
            # The LLM should determine if user is continuing the general conversation or switching topics
            
            # Use LLM to analyze conversation flow
            response = llm.invoke(self.decision_prompt.format(
                current_agent=current_agent,
                last_agent_message=last_agent_message,
                user_message=user_message,
                conversation_history=formatted_history
            ))
            
            # Parse LLM response
            decision_data = self._parse_llm_decision(response.content)
            
            logger.info(f"Conversation flow analysis for session {session_id}: {decision_data}")
            return decision_data
            
        except Exception as e:
            logger.error(f"Error in conversation flow analysis: {str(e)}")
            # Safe fallback - continue with current conversation if there's context
            if current_agent and current_agent != "UNKNOWN":
                return {
                    "decision": "continue",
                    "confidence": 0.6,
                    "reason": f"Error occurred, defaulting to continue with {current_agent}"
                }
            else:
                return {
                    "decision": "switch",
                    "confidence": 0.5,
                    "reason": "Error occurred, defaulting to classify"
                }
    
    def _extract_last_agent_message(self, chat_history: List) -> str:
        """Extract the last message from the agent."""
        try:
            if not chat_history:
                return "No previous message"
            
            # Look for the last agent/AI message
            for item in reversed(chat_history):
                if isinstance(item, dict):
                    if item.get("role") == "assistant":
                        return item.get("content", "")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    author, message = item[0], item[1]
                    if author.lower() in ['ai', 'assistant']:
                        return str(message)
            
            # If no agent message found, get the last message
            last_item = chat_history[-1]
            if isinstance(last_item, dict):
                return last_item.get("content", "No previous message")
            elif isinstance(last_item, (list, tuple)) and len(last_item) >= 2:
                return str(last_item[1])
            
            return "No previous message"
            
        except Exception as e:
            logger.warning(f"Error extracting last agent message: {str(e)}")
            return "No previous message"
    
    def _format_conversation_history(self, chat_history: List, max_exchanges: int = 3) -> str:
        """Format recent conversation history for LLM analysis."""
        try:
            if not chat_history:
                return "No conversation history"
            
            formatted_lines = []
            recent_history = chat_history[-max_exchanges*2:]  # Get last few exchanges
            
            for item in recent_history:
                if isinstance(item, dict):
                    role = item.get("role", "unknown")
                    content = item.get("content", "")
                    speaker = "User" if role == "user" else "Agent"
                    formatted_lines.append(f"{speaker}: {content}")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    author, message = item[0], item[1]
                    speaker = "User" if author.lower() in ['human', 'user'] else "Agent"
                    formatted_lines.append(f"{speaker}: {message}")
            
            return "\n".join(formatted_lines[-6:])  # Limit to last 6 lines
            
        except Exception as e:
            logger.warning(f"Error formatting conversation history: {str(e)}")
            return "No conversation history available"
    
    def _parse_llm_decision(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM JSON response into structured decision."""
        try:
            import json
            # The response is expected to be a JSON string, so we parse it directly.
            decision_data = json.loads(llm_response)
            
            # Validate the parsed data
            if "decision" not in decision_data or "confidence" not in decision_data:
                raise ValueError("Missing required fields in LLM response.")
                
            return decision_data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing LLM decision: {str(e)}")
            # Fallback to a safe default
            return {
                "decision": "continue",
                "confidence": 0.5,
                "reason": "Parse error - defaulting to continue"
            }
    
    def should_continue_conversation(
        self, 
        session_id: str, 
        user_message: str, 
        chat_history: List
    ) -> bool:
        """
        Main method to determine if conversation should continue with current agent.
        Returns True if should continue, False if should reclassify.
        """
        try:
            analysis = self.analyze_conversation_flow(session_id, user_message, chat_history)
            
            decision = analysis["decision"]
            confidence = analysis["confidence"]
            
            # High confidence decisions
            if confidence >= 0.8:
                return decision == "continue"
            
            # Medium confidence - prefer continuation unless explicitly switching
            elif confidence >= 0.5:
                return decision in ["continue", "clarify"]
            
            # Low confidence - be conservative and continue if there's existing context
            else:
                session = get_session(session_id)
                current_agent = session.get("conversation_context", {}).get("primary_product")
                return bool(current_agent and current_agent != "UNKNOWN")
                
        except Exception as e:
            logger.error(f"Error in should_continue_conversation: {str(e)}")
            # Safe fallback - continue if there's existing context
            session = get_session(session_id)
            current_agent = session.get("conversation_context", {}).get("primary_product")
            return bool(current_agent and current_agent != "UNKNOWN")
    
    def get_continuation_reason(
        self, 
        session_id: str, 
        user_message: str, 
        chat_history: List
    ) -> str:
        """Get explanation for why conversation should continue or switch."""
        try:
            analysis = self.analyze_conversation_flow(session_id, user_message, chat_history)
            return analysis.get("reason", "Conversation flow analysis")
        except Exception as e:
            return f"Analysis error: {str(e)}"

# Global instance
conversation_flow_manager = ConversationFlowManager()

def should_continue_with_current_agent(session_id: str, user_message: str, chat_history: List) -> Dict[str, Any]:
    """
    Main function to determine conversation routing based on intelligent flow analysis.
    
    Returns:
        Dict with 'should_continue', 'confidence', and 'reason'
    """
    try:
        should_continue = conversation_flow_manager.should_continue_conversation(
            session_id, user_message, chat_history
        )
        
        reason = conversation_flow_manager.get_continuation_reason(
            session_id, user_message, chat_history
        )
        
        analysis = conversation_flow_manager.analyze_conversation_flow(
            session_id, user_message, chat_history
        )
        
        return {
            "should_continue": should_continue,
            "confidence": analysis.get("confidence", 0.7),
            "reason": reason,
            "decision": analysis.get("decision", "continue")
        }
        
    except Exception as e:
        logger.error(f"Error in conversation routing analysis: {str(e)}")
        return {
            "should_continue": True,  # Conservative fallback
            "confidence": 0.5,
            "reason": f"Error occurred: {str(e)}",
            "decision": "continue"
        }
