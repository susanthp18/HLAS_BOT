import os
import logging
from typing import Dict, List, Any, Optional
from app.session_manager import (
    get_session, update_session, get_chat_history, get_stage, set_stage,
    update_conversation_context, increment_error_count, set_collected_info,
    conversations_collection
)
from .primary_intent_agent import get_primary_intent, Product, validate_user_input
from .conversation_flow_manager import should_continue_with_current_agent
from .travel_agent import run_travel_agent
# from .maid_agent import run_maid_agent
from .payment_agent import run_payment_agent
from .fallback_system import get_fallback_response, handle_agent_failure, detect_confusion
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from app.config import llm
from pydantic import BaseModel, Field
from typing import List
from .rec_retriever_agent import get_available_tiers, generate_plan_comparison_table

logger = logging.getLogger(__name__)

class TierExtraction(BaseModel):
    tiers: List[str] = Field(default=[], description="A list of insurance plan tiers mentioned by the user for comparison, e.g., ['Gold', 'Silver']. Should be an empty list if no specific tiers are mentioned.")

def extract_comparison_tiers(user_message: str, product: str, chat_history: list, session_id: str) -> List[str]:
    """
    Intelligently extracts which plan tiers a user wants to compare.
    """
    logger.info(f"Extracting comparison tiers from user message: '{user_message}'")
    available_tiers = get_available_tiers(product)
    
    # Get the previously recommended plan from session context if it exists
    session = get_session(session_id)
    recommended_plan = session.get("conversation_context", {}).get("recommended_plan")
    
    system_prompt = f"""You are an expert at extracting specific entities from a user's message in the context of a conversation.
The user is talking about {product} insurance and wants to compare plan tiers.
The available tiers are: {', '.join(available_tiers)}.

- Analyze the user's message and the conversation history.
- Identify any specific plan tiers the user mentions.
- If the user uses vague terms like "this one", "that one", or "it", and a plan was just recommended, you should assume they are referring to the recommended plan. The previously recommended plan was: '{recommended_plan}'.
- Return a list of the tier names. If they don't mention any specific tiers, return an empty list.

Conversation History (most recent messages):
{chat_history[-4:]}
"""
    
    chain = llm.with_structured_output(TierExtraction, method="function_calling")
    
    try:
        response = chain.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User's request: {user_message}")
        ])
        logger.info(f"Extracted tiers for comparison: {response.tiers}")
        return response.tiers
    except Exception as e:
        logger.error(f"Failed to extract comparison tiers: {e}")
        return [] # Return empty list on failure

def handle_unknown_product_intelligently(user_message: str, chat_history: list, session_id: str) -> str:
    """
    LLM-powered intelligent routing for UNKNOWN product states.
    Uses context and conversation analysis instead of keyword matching.
    """
    try:
        # Use the primary intent agent to classify the user's message
        intent_result = get_primary_intent(user_message, chat_history)
        
        # If a specific product is detected, route to that agent
        if hasattr(intent_result, 'product') and intent_result.product != Product.UNKNOWN:
            if intent_result.product == Product.TRAVEL:
                logger.info(f"LLM routing to TRAVEL agent for session {session_id}")
                update_conversation_context(session_id, primary_product=Product.TRAVEL, last_intent="product_inquiry")
                set_stage(session_id, "travel_inquiry")
                travel_response = run_travel_agent(user_message, chat_history, session_id)
                if isinstance(travel_response, dict):
                    return travel_response.get("output", "I'd be happy to help with travel insurance! 🌍✈️")
                else:
                    return travel_response
                    
            # elif intent_result.product == Product.MAID:
            #     logger.info(f"LLM routing to MAID agent for session {session_id}")
            #     update_conversation_context(session_id, primary_product=Product.MAID, last_intent="product_inquiry")
            #     set_stage(session_id, "maid_inquiry")
            #     maid_response = run_maid_agent(user_message, chat_history, session_id)
            #     if isinstance(maid_response, dict):
            #         return maid_response.get("output", "I'd be happy to help with maid insurance! 🏠")
            #     else:
            #         return maid_response
        
        # Handle different intents based on LLM classification
        if hasattr(intent_result, 'intent'):
            if intent_result.intent == "payment_inquiry":
                # Check if user has gone through the recommendation process
                session = get_session(session_id)
                collected_info = session.get('collected_info', {})
                
                if collected_info:
                    logger.info(f"LLM routing to PAYMENT agent for session {session_id}")
                    set_stage(session_id, "payment")
                    payment_response = run_payment_agent(user_message, chat_history, session_id)
                    if isinstance(payment_response, dict):
                        return payment_response.get("output", "I'll help you with the payment process! 💳")
                    else:
                        return payment_response
                else:
                    return """I'd love to help you with payment! 💳

But first, let me recommend the perfect insurance plan for you. Which type are you interested in?

🌍 **Travel Insurance** - for your trips and vacations
"""
# 🏠 **Maid Insurance** - for your domestic helper
# 
# Once I recommend the best plan with all coverage details, we can proceed with payment! 😊"""
                    
            elif intent_result.intent == "product_inquiry":
                # General product inquiry - guide to product selection
                return """Great! I'd be happy to help you get the perfect insurance! 😊

To recommend the best plan for you, I need to know what type of insurance you're interested in:

🌍 **Travel Insurance** - for your trips and vacations
"""
# 🏠 **Maid Insurance** - for your domestic helper
# 
# Which type would you like to explore? Once you choose, I'll ask a few quick questions and recommend the perfect plan with all the coverage details! ✨"""
                
            elif intent_result.intent == "informational":
                # Handle informational questions when product is unknown
                logger.info(f"🎯 UNKNOWN HANDLER: Informational question without product - setting stage to 'awaiting_product_for_rag'")
                set_stage(session_id, "awaiting_product_for_rag")
                update_conversation_context(session_id, pending_rag_question=user_message)
                
                return """I can help with that.\n\nWhich insurance?\n• Travel Insurance"""
        
        # Default to intelligent guidance
        return provide_intelligent_guidance(user_message, chat_history, session_id)
            
    except Exception as e:
        logger.error(f"Error in LLM-powered agent routing: {str(e)}")
        return "I'm here to help! Could you tell me more about what insurance you're interested in? 😊"

def provide_intelligent_guidance(user_message: str, chat_history: list, session_id: str) -> str:
    """
    Provide intelligent guidance when user intent is unclear.
    Uses conversation context to give helpful responses.
    """
    try:
        # Analyze conversation context for intelligent guidance
        if len(chat_history) <= 2:
            # Early in conversation - provide welcoming guidance
            # return "Great! I can help you with Travel insurance for your trips or Maid insurance for your domestic helper. Which would you like to learn about? ✈️🏠"
            return "Great! I can help you with Travel insurance for your trips. Which would you like to learn about? ✈️"
        else:
            # Later in conversation - provide contextual guidance
            return "I want to make sure I help you with the right information! Are you interested in Travel insurance for your trips? Just let me know! 😊"
            
    except Exception as e:
        logger.error(f"Error providing intelligent guidance: {str(e)}")
        return "I'm here to help with your insurance needs! What can I assist you with today? 😊"

def get_whatsapp_fallback_response(intent_type: str, requires_clarification: bool = False) -> str:
    """
    Generate appropriate WhatsApp fallback responses for different scenarios.
    """
    responses = {
        "invalid_input": [
            "I didn't quite catch that! 😅 Could you please rephrase your question about insurance?",
            "Hmm, I'm not sure I understand. Could you tell me what insurance information you're looking for? 🤔",
            "Sorry, I couldn't understand that message. How can I help you with your insurance needs today? 💬"
        ],
        "clarification_needed": [
            "I'd love to help! Could you be a bit more specific about what insurance information you need? 🙂",
            "I want to make sure I give you the right information. Could you tell me more about what you're looking for? 📋",
            "To better assist you, could you please clarify what type of insurance you're interested in? 🤝"
        ],
        "off_topic": [
            # "I specialize in helping with insurance questions! 😊 What can I help you with regarding Travel or Maid insurance?",
            "I specialize in helping with insurance questions! 😊 What can I help you with regarding Travel insurance?",
            # "I'm here to assist with your insurance needs. How can I help with Travel or Maid insurance today? 🛡️",
            "I'm here to assist with your insurance needs. How can I help with Travel insurance today? 🛡️",
            # "Let's talk insurance! I can help you with Travel or Maid insurance. What would you like to know? ✈️🏠"
            "Let's talk insurance! I can help you with Travel insurance. What would you like to know? ✈️"
        ],
        "error": [
            "Oops! Something went wrong on my end. 😅 Could you please try again?",
            "I'm having a small technical hiccup. Please send your message again! 🔧",
            "Sorry about that! There was a brief issue. Could you resend your question? 💻"
        ]
    }
    
    import random
    return random.choice(responses.get(intent_type, responses["error"]))

def handle_low_confidence_intent(intent_result, user_message: str, chat_history: list) -> str:
    """
    Handle cases where the intent classification has low confidence.
    """
    if intent_result.confidence < 0.4:
        logger.warning(f"Very low confidence intent: {intent_result.confidence}")
        return get_whatsapp_fallback_response("clarification_needed")
    
    elif intent_result.confidence < 0.6:
        # Try to provide some guidance based on partial understanding
        if intent_result.product != Product.UNKNOWN:
            return f"I think you're asking about {intent_result.product.value.lower()} insurance! 🤔 Could you tell me a bit more about what specific information you need?"
        else:
            return get_whatsapp_fallback_response("clarification_needed")
    
    return None  # Continue with normal processing

def get_contextual_greeting() -> str:
    """
    Generate a contextual greeting for WhatsApp.
    """
    import datetime
    hour = datetime.datetime.now().hour
    
    if 5 <= hour < 12:
        time_greeting = "Good morning"
    elif 12 <= hour < 17:
        time_greeting = "Good afternoon"
    elif 17 <= hour < 21:
        time_greeting = "Good evening"
    else:
        time_greeting = "Hello"
    
    greetings = [
        # f"{time_greeting}! 😊 I'm here to help you with Travel and Maid insurance. What can I assist you with today?",
        f"{time_greeting}! 😊 I'm here to help you with Travel insurance. What can I assist you with today?",
        # f"{time_greeting}! 👋 Welcome to HLAS! I can help you with Travel or Maid insurance. How may I help you?",
        f"{time_greeting}! 👋 Welcome to HLAS! I can help you with Travel insurance. How may I help you?",
        # f"{time_greeting}! 🌟 I'm your insurance assistant. I specialize in Travel and Maid insurance. What would you like to know?"
        f"{time_greeting}! 🌟 I'm your insurance assistant. I specialize in Travel insurance. What would you like to know?"
    ]
    
    import random
    return random.choice(greetings)

def orchestrate_chat(user_message: str, session_id: str) -> str:
    """
    Enhanced central router for directing user messages to the appropriate agent with 
    comprehensive error handling and intelligent conversation management.

    Args:
        user_message: The message from the user.
        session_id: A unique identifier for the conversation session.

    Returns:
        The response from the appropriate agent.
    """
    try:
        logger.info(f"Processing message from session {session_id}: {user_message[:100]}...")
        
        # Validate input first
        validation_result = validate_user_input(user_message)
        if not validation_result["is_valid"]:
            logger.warning(f"Invalid input from session {session_id}: {validation_result['issue_type']}")
            update_session(session_id, user_message, validation_result["message"])
            return validation_result["message"]
        
        chat_history = get_chat_history(session_id)
        stage = get_stage(session_id)
        session = get_session(session_id)
        
        # Debug logging for stage
        logger.info(f"Current stage for session {session_id}: {stage}")

        # Handle payment stage with enhanced error handling
        if stage == "payment":
            try:
                logger.info(f"Processing payment stage message for session {session_id}")
                response_data = run_payment_agent(user_message, chat_history, session_id)
                
                # Check if the payment agent has signaled a context switch
                if response_data.get("requires_context_switch"):
                    logger.warning(f"Payment agent signaled a context switch for session {session_id}. Resetting flow.")
                    
                    # 1. Clear all collected info for a completely fresh start.
                    conversations_collection.update_one(
                        {"session_id": session_id},
                        {"$set": {"collected_info": {}}}
                    )
                    logger.info(f"Cleared all collected_info for session {session_id} on context switch.")
                    
                    # 2. Reset the stage.
                    set_stage(session_id, "initial")
                    
                    # 3. Re-route the original message through the main orchestrator logic to handle the new intent.
                    # We call orchestrate_chat again to restart the process cleanly.
                    return orchestrate_chat(user_message, session_id)

                if isinstance(response_data, dict):
                    agent_response = response_data.get("output", "I'm processing your payment information. Please provide your details.")
                else:
                    agent_response = str(response_data)
                    
                # Check if payment is complete and reset stage
                if "payment gateway" in agent_response.lower() or "redirected" in agent_response.lower():
                    set_stage(session_id, "initial")
                    agent_response += "\n\nFeel free to ask me any other insurance questions! 😊"
                    
            except Exception as e:
                logger.error(f"Payment agent error for session {session_id}: {str(e)}")
                agent_response = "I'm experiencing a technical issue with payment processing. 😅 Please try again or contact our support team for assistance."
                set_stage(session_id, "initial")  # Reset stage on error
                
        elif stage == "awaiting_product_for_rag":
            logger.info(f"🎯 STAGE HANDLER: Processing 'awaiting_product_for_rag' stage for session {session_id}")
            product_intent = get_primary_intent(user_message, chat_history)
            
            if hasattr(product_intent, 'product') and product_intent.product != Product.UNKNOWN:
                # Product has been identified, answer the pending question
                conversation_context = session.get("conversation_context", {})
                pending_question = conversation_context.get("pending_rag_question")
                if pending_question:
                    logger.info(f"Product '{product_intent.product.value}' identified. Answering pending question: '{pending_question}'")
                    from .rag_agent import get_rag_response
                    agent_response = get_rag_response(pending_question, chat_history, product_intent.product.value)
                    
                    # Reset stage and clear pending question
                    set_stage(session_id, "initial")
                    update_conversation_context(session_id, pending_rag_question=None)
                else:
                    # This case should not happen, but as a fallback, route to the agent
                    logger.warning(f"In 'awaiting_product_for_rag' stage but no pending question found. Routing to agent.")
                    set_stage(session_id, "initial") # Reset stage
                    agent_response = process_normal_intent(product_intent, user_message, chat_history, session_id)
            else:
                # Product still not identified, re-prompt
                logger.info(f"Product still not identified. Re-prompting.")
                agent_response = """Which insurance?\n• Travel Insurance"""
                
        # Handle recommendation stage - LLM-powered decision making
        elif stage == "recommendation":
            try:
                logger.info(f"Processing recommendation stage message for session {session_id}")
                
                # Use LLM to understand user intent in recommendation stage
                
                class RecommendationStageIntent(BaseModel):
                    intent: str = Field(..., description="User's intent: 'purchase' (wants to buy/proceed), 'plan_comparison' (wants different/other plans), 'policy_question' (has questions about coverage), or 'other'")
                    confidence: float = Field(..., description="Confidence in classification (0.0 to 1.0)")
                    reasoning: str = Field(..., description="Brief explanation of the classification")
                
                # Get current context
                session = get_session(session_id)
                current_product = session.get("conversation_context", {}).get("primary_product", "UNKNOWN")
                
                # LLM-powered intent classification for recommendation stage
                intent_chain = llm.with_structured_output(RecommendationStageIntent, method="function_calling")
                
                intent_prompt = [
                    SystemMessage(content=f"""You are analyzing user intent in the recommendation stage of {current_product} insurance conversation.

The user has just received a plan recommendation with benefits and options to:
1. Ask about different plan tiers (higher/lower)
2. Ask questions about coverage/benefits  
3. Proceed to purchase
4. Other requests

Analyze the user's message and conversation history to determine their intent.

**CRITICAL**: If the user's message explicitly mentions a different insurance product (e.g., 'maid' when the current product is TRAVEL), you MUST classify the intent as 'other' to trigger a re-routing.

Chat History:
{chat_history[-6:] if len(chat_history) > 6 else chat_history}"""),
                    HumanMessage(content=f"User message: {user_message}")
                ]
                
                intent_result = intent_chain.invoke(intent_prompt)
                logger.info(f"Recommendation stage intent: {intent_result.intent} (confidence: {intent_result.confidence})")
                
                # UNIVERSAL CHECK: Always re-classify with the primary agent to catch any context switches first.
                primary_intent_result = get_primary_intent(user_message, chat_history)

                if primary_intent_result.product != Product.UNKNOWN and primary_intent_result.product != current_product:
                    # This is a product switch. Handle it regardless of the recommendation stage intent.
                    logger.info(f"Detected product switch during recommendation stage from {current_product} to {primary_intent_result.product}.")
                    
                    # Clear all collected info for a fresh start on product switch.
                    conversations_collection.update_one(
                        {"session_id": session_id},
                        {"$set": {"collected_info": {}}}
                    )
                    logger.info(f"Cleared all collected_info for session {session_id} on product switch")

                    # Update context and route to the new flow
                    update_conversation_context(session_id, primary_product=primary_intent_result.product, last_intent="product_inquiry")
                    set_stage(session_id, "initial")
                    agent_response = process_normal_intent(primary_intent_result, user_message, chat_history, session_id)
                else:
                    # No product switch detected, proceed with routing based on recommendation stage intent.
                    if intent_result.intent == "purchase" and intent_result.confidence > 0.6:
                        logger.info(f"User wants to proceed to purchase for session {session_id}")
                        set_stage(session_id, "payment")
                        agent_response = run_payment_agent(user_message, chat_history, session_id)
                        if isinstance(agent_response, dict):
                            agent_response = agent_response.get("output", "Let me help you with the payment process! 💳")
                    
                    elif intent_result.intent == "plan_comparison" and intent_result.confidence > 0.6:
                        logger.info(f"User asking for plan comparison for session {session_id}")
                        
                        # Convert product enum to string for collection lookup
                        product_str = current_product.value if hasattr(current_product, 'value') else str(current_product)

                        # Intelligently extract which tiers the user wants to compare
                        requested_tiers = extract_comparison_tiers(user_message, product_str, chat_history, session_id)
                        
                        # Generate the full comparison table, highlighting the requested tiers
                        agent_response = generate_plan_comparison_table(product_str, requested_tiers)

                        # Add a follow-up prompt
                        agent_response += "\n\nWould you like to proceed with one of these plans or do you have other questions?"
                    
                    elif intent_result.intent == "other" and intent_result.confidence > 0.6:
                        logger.info(f"Recommendation stage intent is 'other'. Already checked for product switch. Now checking for special intents.")
                        
                        # Case 1: User is asking about claim status (already handled by the universal check if product was different)
                        if primary_intent_result.intent == "policy_claim_status":
                            logger.info(f"Detected policy/claim status check. Providing stubbed response.")
                            agent_response = """*Policy/Claim Status Check*\n\nCurrently under development.\n\nWill require NRIC number when available.\n\nCan I help with:\n• Coverage questions\n• Benefits information\n• New insurance purchase"""
                        
                        # Case 2: User is asking for a live agent
                        elif primary_intent_result.intent == "connect_to_live_agent":
                            logger.info(f"Detected live agent request in recommendation stage. Providing placeholder.")
                            update_conversation_context(session_id, live_agent_request=True, last_intent="connect_to_live_agent")
                            set_stage(session_id, "live_agent")
                            agent_response = "I'll connect you to our live agent shortly. Please wait while we transfer you to a specialist who can assist you better! 👥"

                        # Case 3: Fallback to RAG if it's not a clear special intent
                        else:
                            logger.info(f"Re-classification did not yield a clear action. Treating as policy question for current product: {current_product}")
                            from .rag_agent import get_rag_response
                            product_str = current_product.value if hasattr(current_product, 'value') else str(current_product)
                            agent_response = get_rag_response(user_message, chat_history, product_str)
                            agent_response += "\n\nDo you have any other questions about the coverage, or would you like to proceed with purchasing this plan?"
                    
                    else:  # This now primarily handles 'policy_question'
                        logger.info(f"User asking policy question about {current_product} for session {session_id}")
                        from .rag_agent import get_rag_response
                        # Convert product enum to string for collection lookup
                        product_str = current_product.value if hasattr(current_product, 'value') else str(current_product)
                        agent_response = get_rag_response(user_message, chat_history, product_str)
                        agent_response += "\n\nDo you have any other questions about the coverage, or would you like to proceed with purchasing this plan?"
                    
            except Exception as e:
                logger.error(f"Recommendation stage error for session {session_id}: {str(e)}")
                agent_response = "I'm having a technical issue. 😅 Could you please rephrase your question or let me know if you'd like to proceed with the purchase?"
                
        elif stage == "travel_inquiry":
            # If we are in the data collection flow for travel, all messages should go to the travel agent.
            # This prevents ambiguous answers from being mis-routed to RAG.
            logger.info(f"Continuing dedicated 'travel_inquiry' flow for session {session_id}.")
            response_data = run_travel_agent(user_message, chat_history, session_id)
            if isinstance(response_data, dict):
                agent_response = response_data.get("output", "I'm processing your travel details. One moment please.")
            else:
                agent_response = str(response_data)

        else:
            # *** ROBUST LLM-POWERED CONVERSATION FLOW ROUTING ***
            # Use intelligent conversation flow analysis
            flow_analysis = should_continue_with_current_agent(session_id, user_message, chat_history)
            
            should_continue = flow_analysis["should_continue"]
            flow_confidence = flow_analysis["confidence"]
            flow_reason = flow_analysis["reason"]
            
            logger.info(f"Conversation flow analysis: {flow_analysis}")
            
            if should_continue:
                logger.info(f"🔄 CONTINUING conversation for session {session_id}")
                # Get the real intent for the current message to route correctly.
                intent_result = get_primary_intent(user_message, chat_history)
                
                # Prioritize the existing product context from the session.
                current_session = get_session(session_id)
                context = current_session.get("conversation_context", {})
                current_product = context.get("primary_product")

                if current_product and current_product != Product.UNKNOWN:
                    # If product context exists, override the classification to maintain context.
                    logger.info(f"Maintaining product context: {current_product}. Using new intent: {intent_result.intent}")
                    intent_result.product = current_product if isinstance(current_product, Product) else Product(current_product.upper())
                else:
                    # No existing product context, so we trust the new classification.
                    logger.info(f"No prior product context. Using new classification: {intent_result.product}")
                    update_conversation_context(session_id, primary_product=intent_result.product)

                product = intent_result.product
                intent = intent_result.intent
                
                logger.info(f"CONVERSATION CONTINUATION - Product: {product}, Intent: {intent}, Reason: {flow_reason}")
                
            else:
                logger.info(f"🆕 NEW CLASSIFICATION for session {session_id}")
                # Classify new message or topic switch
                intent_result = get_primary_intent(user_message, chat_history)
                product = intent_result.product
                intent = intent_result.intent
                confidence = intent_result.confidence
                requires_clarification = intent_result.requires_clarification
                
                logger.info(f"NEW CLASSIFICATION - Product: {product}, Intent: {intent}, Confidence: {confidence}")
                
                # Update conversation context with new classification
                update_conversation_context(session_id, primary_product=product, last_intent=intent)
            
            # Handle invalid input or low confidence first
            if intent == "invalid_input":
                logger.info(f"❌ INVALID INPUT for session {session_id}")
                agent_response = get_fallback_response("input_validation_error", session_id)
            elif hasattr(intent_result, 'requires_clarification') and (intent_result.requires_clarification or intent_result.confidence < 0.6):
                logger.info(f"🔍 LOW CONFIDENCE check - requires_clarification: {intent_result.requires_clarification}, confidence: {intent_result.confidence}")
                low_conf_response = handle_low_confidence_intent(intent_result, user_message, chat_history)
                if low_conf_response:
                    logger.info(f"📝 Using low confidence response")
                    agent_response = low_conf_response
                else:
                    # Continue with normal processing for medium confidence
                    logger.info(f"📝 Calling process_normal_intent for medium confidence")
                    agent_response = process_normal_intent(intent_result, user_message, chat_history, session_id)
            elif intent == "greeting":
                logger.info(f"👋 GREETING handler for session {session_id}")
                # Update conversation context
                update_conversation_context(session_id, has_greeted=True)
                agent_response = get_contextual_greeting()
            elif intent == "clarification_needed":
                agent_response = get_fallback_response("input_validation_error", session_id)
            elif intent == "other":
                agent_response = get_fallback_response("off_topic", session_id)
            elif intent == "informational":
                # Handle informational questions
                logger.info(f"🔍 MAIN HANDLER: Processing informational question for session {session_id}")
                
                # Check if product is specified
                if hasattr(intent_result, 'product') and intent_result.product != Product.UNKNOWN:
                    # Product specified - route to RAG with specific product
                    logger.info(f"Routing informational question to RAG agent for {intent_result.product.value} product")
                    from .rag_agent import get_rag_response
                    agent_response = get_rag_response(user_message, chat_history, intent_result.product.value)
                else:
                    # Product not specified - set stage and ask user to choose
                    logger.info(f"🎯 MAIN HANDLER: Informational question without product - setting stage to 'awaiting_product_for_rag'")
                    set_stage(session_id, "awaiting_product_for_rag")
                    update_conversation_context(session_id, pending_rag_question=user_message)
                    
                    agent_response = """I can help with that.\n\nWhich insurance?\n• Travel Insurance"""
            elif intent == "policy_claim_status":
                # Handle actual policy/claim status checks (requires NRIC)
                logger.info(f"🔍 MAIN HANDLER: Processing policy/claim status check for session {session_id}")
                agent_response = """*Policy/Claim Status Check*\n\nCurrently under development.\n\nWill require NRIC number when available.\n\nCan I help with:\n• Coverage questions\n• Benefits information\n• New insurance purchase"""
            elif intent == "connect_to_live_agent":
                logger.info(f"User wants to talk to live agent for session {session_id}")
                update_conversation_context(session_id, live_agent_request=True, last_intent="connect_to_live_agent")
                set_stage(session_id, "live_agent")
                agent_response = "I'll connect you to our live agent shortly. Please wait while we transfer you to a specialist who can assist you better! 👥"
            else:
                # Skip confusion detection during data collection stages for travel/maid agents,
                # as it can misinterpret valid contextual inputs like dates or numbers.
                # is_data_collection_stage = should_continue and product in [Product.TRAVEL, Product.MAID]
                is_data_collection_stage = should_continue and product in [Product.TRAVEL]
                confusion_response = None
                if not is_data_collection_stage:
                    confusion_response = detect_confusion(session_id, user_message)

                if confusion_response:
                    logger.info(f"🤔 CONFUSION detected for session {session_id}")
                    agent_response = confusion_response
                # *** INTELLIGENT AGENT-BASED ROUTING FOR UNKNOWN PRODUCTS ***
                # If continuing with UNKNOWN, use intelligent content-based routing
                elif should_continue and product == Product.UNKNOWN:
                    agent_response = handle_unknown_product_intelligently(user_message, chat_history, session_id)
                else:
                    agent_response = process_normal_intent(intent_result, user_message, chat_history, session_id)
            
            # Update conversation context with current intent
            update_conversation_context(session_id, last_intent=intent, primary_product=product)

        # Update chat history
        update_session(session_id, user_message, agent_response)
        
        logger.info(f"Response generated for session {session_id}: {len(agent_response)} characters")
        return agent_response
        
    except Exception as e:
        logger.error(f"Critical error in orchestrate_chat for session {session_id}: {str(e)}")
        error_response = get_fallback_response("general_error", session_id)
        
        # Still try to update session even on error
        try:
            update_session(session_id, user_message, error_response)
        except:
            pass  # Don't let session update failure break the response
            
        return error_response

def process_normal_intent(intent_result, user_message: str, chat_history: list, session_id: str) -> str:
    """
    Process normal intents with proper error handling and agent routing.
    """
    try:
        product = intent_result.product
        intent = intent_result.intent
        
        agent_map = {
            Product.TRAVEL: run_travel_agent,
            # Product.MAID: run_maid_agent,
        }

        if intent == "payment_inquiry":
            logger.info(f"Setting session {session_id} to payment stage")
            set_stage(session_id, "payment")
            try:
                response_data = run_payment_agent(user_message, chat_history)
                if isinstance(response_data, dict):
                    return response_data.get("output", "Let me help you with the payment process! 💳")
                else:
                    return str(response_data)
            except Exception as e:
                logger.error(f"Payment agent initialization error: {str(e)}")
                return handle_agent_failure(session_id, "payment_agent", str(e))
                
        elif product in agent_map:
            try:
                agent_function = agent_map[product]
                logger.info(f"Routing to {product.value} agent for session {session_id}")
                set_stage(session_id, f"{product.value.lower()}_inquiry")
                
                # Pass session_id to agents that support it
                if product == Product.TRAVEL:
                    response_data = agent_function(user_message, chat_history, session_id)
                # elif product == Product.MAID:
                #     response_data = agent_function(user_message, chat_history, session_id)
                else:
                    response_data = agent_function(user_message, chat_history)
                
                if isinstance(response_data, dict):
                    return response_data.get("output", f"I'm here to help with {product.value.lower()} insurance! 😊 What specific information do you need?")
                else:
                    return str(response_data)
                    
            except Exception as e:
                logger.error(f"{product.value} agent error: {str(e)}")
                return handle_agent_failure(session_id, f"{product.value.lower()}_agent", str(e))
                
        else:
            # Enhanced unknown product handling with context awareness
            return get_fallback_response("product_not_available", session_id)
                
    except Exception as e:
        logger.error(f"Error in process_normal_intent: {str(e)}")
        return get_fallback_response("general_error", session_id)
