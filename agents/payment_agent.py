import os
import re
import logging
import requests
import json
from pydantic import BaseModel, Field, validator
from app.config import llm
from langchain_core.messages import SystemMessage, HumanMessage
from app.session_manager import get_session, set_collected_info, get_collected_info
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class PaymentStage(BaseModel):
    """Model for payment stage analysis."""
    stage: str = Field(..., description="Current payment stage: 'plan_confirmation', 'collecting_details', 'processing_payment', 'completed'")
    user_intent: str = Field(..., description="User's intent: 'confirm_plan', 'provide_details', 'cancel', 'question'")
    extracted_name: Optional[str] = Field(None, description="Extracted full name if provided")
    extracted_email: Optional[str] = Field(None, description="Extracted email if provided")
    confidence: float = Field(..., description="Confidence in analysis (0.0 to 1.0)")
    response: str = Field(..., description="Response to user")

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_name(name: str) -> bool:
    """Validate name format."""
    return len(name.strip()) >= 2 and all(part.replace(' ', '').replace('-', '').replace("'", '').isalpha() for part in name.strip().split())

# COMMENTED OUT - Not storing payment details
# def store_user_plan(name: str, email: str, product: str, plan_name: str) -> bool:
#     """Store user plan data to MongoDB via API."""
#     try:
#         api_url = "http://localhost:3000/api/user_plans"
#         data = {
#             "name": name,
#             "mail_id": email,
#             "product": product,
#             "plan_name": plan_name
#         }
#         
#         response = requests.post(api_url, json=data, timeout=10)
#         if response.status_code == 200 or response.status_code == 201:
#             logger.info(f"Successfully stored user plan for {name}")
#             return True
#         else:
#             logger.error(f"Failed to store user plan: {response.status_code} - {response.text}")
#             return False
#             
#     except Exception as e:
#         logger.error(f"Error storing user plan: {str(e)}")
#         return False

def run_payment_agent(user_message: str, chat_history: list, session_id: str) -> Dict[str, Any]:
    """
    LLM-powered payment agent that handles plan confirmation and payment processing.
    """
    try:
        session = get_session(session_id)
        collected_info = session.get("collected_info", {})
        conversation_context = session.get("conversation_context", {})
        
        # Get recommended plan info
        recommended_plan = conversation_context.get("recommended_plan", "Standard")
        primary_product = conversation_context.get("primary_product")
        
        if not primary_product:
            return {
                "output": "I need to know which insurance product you're interested in purchasing. Please let me know if it's Travel or Maid insurance.",
                "stage": "error"
            }
        
        # Convert enum to string if needed
        product_name = primary_product.value if hasattr(primary_product, 'value') else str(primary_product)
        
        # Get current payment info
        payment_info = collected_info.get("payment_info", {})
        
        # Use LLM to analyze payment stage and user intent
        try:
            payment_chain = llm.with_structured_output(PaymentStage, method="json_mode")
    
            prompt = [
                SystemMessage(content=f"""You are a payment processing assistant for HLAS Insurance. 

CONTEXT:
- Product: {product_name} Insurance
- Recommended Plan: {recommended_plan}
- Current payment info collected: {payment_info}
- Chat history: {chat_history[-4:] if len(chat_history) > 4 else chat_history}

PAYMENT STAGES:
1. 'plan_confirmation' - User needs to confirm they want to purchase the recommended plan
2. 'collecting_details' - Collecting name and email for payment processing  
3. 'processing_payment' - Ready to process payment with all details
4. 'completed' - Payment process completed

USER INTENTS:
- 'confirm_plan' - User confirms they want to buy the plan
- 'provide_details' - User is providing name/email details
- 'cancel' - User wants to cancel or go back
- 'question' - User has questions about the plan/payment

VALIDATION RULES:
- Name: Must be at least 2 characters, only letters, spaces, hyphens, apostrophes
- Email: Must be valid email format (user@domain.com)

INSTRUCTIONS:
1. Analyze the user's message in the context of the payment flow.
2. Determine the current `stage`, the `user_intent`, and your `confidence` level (as a number between 0.0 and 1.0).
3. Formulate a `response` to the user based on the current stage and their intent.
4. If the user provides their name or email, extract it into `extracted_name` or `extracted_email`.
5. Your final output must be a JSON object containing all required fields: `stage`, `user_intent`, `confidence` (as float), and `response`.

IMPORTANT: `confidence` must be a number between 0.0 and 1.0 (e.g., 0.95, 0.8, 0.7), NOT a word like 'high' or 'low'.
"""),
                HumanMessage(content=f"User message: {user_message}")
            ]
            
            result = payment_chain.invoke(prompt)
            
            # Fallback: Fix confidence if it's not a valid float
            if not isinstance(result.confidence, (int, float)) or not (0.0 <= result.confidence <= 1.0):
                logger.warning(f"Invalid confidence value: {result.confidence}, setting to 0.8")
                result.confidence = 0.8
            
            logger.info(f"Payment stage analysis: {result.stage}, Intent: {result.user_intent}, Confidence: {result.confidence}")

            # Fallback: If LLM fails to extract email, try regex
            if not result.extracted_email:
                email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', user_message)
                if email_match:
                    extracted = email_match.group(0)
                    if validate_email(extracted):
                        logger.info(f"Regex fallback extracted a valid email: {extracted}")
                        result.extracted_email = extracted
            
            # Fallback: If LLM fails to extract name, try common patterns
            if not result.extracted_name:
                # Look for patterns like "my name is X", "I am X", "name: X", etc.
                name_patterns = [
                    r'(?:my name is|i am|name is|name:)\s+([a-zA-Z\s\'-]{2,30})',
                    r'([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+)*)',  # Capitalized names
                ]
                
                for pattern in name_patterns:
                    name_match = re.search(pattern, user_message, re.IGNORECASE)
                    if name_match:
                        extracted_name = name_match.group(1).strip()
                        if validate_name(extracted_name) and len(extracted_name.split()) <= 4:  # Reasonable name length
                            logger.info(f"Regex fallback extracted a valid name: {extracted_name}")
                            result.extracted_name = extracted_name
                            break
            
        except Exception as e:
            logger.error(f"Error in payment LLM analysis: {str(e)}")
            return {
                "output": "I'm having trouble processing your payment request. Please try again or contact support.",
                "stage": "error"
            }
        
        # Handle different payment stages
        if result.stage == "plan_confirmation":
            if result.user_intent == "confirm_plan":
                # User confirmed plan, move to details collection
                return {
                    "output": f"*{recommended_plan} {product_name} Insurance* - Confirmed\n\nRequired details:\n• Full name\n• Email address\n\nPlease provide your full name first.",
                    "stage": "collecting_details"
                }
            else:
                # Ask for plan confirmation
                return {
                    "output": f"*{product_name} Insurance Purchase*\n\nRecommended: *{recommended_plan} Plan*\n\nConfirm purchase?\n• Type 'yes' to proceed\n• Type 'no' to cancel",
                    "stage": "plan_confirmation"
                }
        
        elif result.stage == "collecting_details":
            # Update collected details
            if result.extracted_name:
                if validate_name(result.extracted_name):
                    payment_info["name"] = result.extracted_name.strip()
                    logger.info(f"Valid name collected: {result.extracted_name}")
                else:
                    return {
                        "output": "*Invalid name format*\n\nRequired:\n• At least 2 characters\n• Letters only\n\nExample: John Smith",
                        "stage": "collecting_details"
                    }
            
            if result.extracted_email:
                if validate_email(result.extracted_email):
                    payment_info["email"] = result.extracted_email.lower().strip()
                    logger.info(f"Valid email collected: {result.extracted_email}")
                else:
                    return {
                        "output": "*Invalid email format*\n\nExample: john.smith@email.com",
                        "stage": "collecting_details"
                    }
            
            # Save updated payment info (temporarily for conversation flow, not permanent storage)
            collected_info["payment_info"] = payment_info  # Keep for conversation flow
            set_collected_info(session_id, "payment_info", payment_info)  # Keep for conversation flow
            
            # Check what's still needed
            if "name" not in payment_info:
                return {
                    "output": "Please provide your full name:",
                    "stage": "collecting_details"
                }
            elif "email" not in payment_info:
                return {
                    "output": f"Thank you, *{payment_info['name']}*\n\nNow provide your email address:",
                    "stage": "collecting_details"
                }
            else:
                # All details collected, process payment
                return process_payment(session_id, payment_info, product_name, recommended_plan)
        
        elif result.stage == "processing_payment":
            # All details should be collected, process payment
            if "name" in payment_info and "email" in payment_info:
                return process_payment(session_id, payment_info, product_name, recommended_plan)
            else:
                return {
                    "output": "I need your name and email address to process the payment. Please provide them.",
                    "stage": "collecting_details"
                }
        
        elif result.user_intent == "cancel":
            return {
                "output": "No problem! If you change your mind about purchasing insurance, just let me know. I'm here to help! 😊",
                "stage": "cancelled"
            }
        
        else:
            # Default response
            return {
                "output": result.response,
                "stage": result.stage
            }
            
    except Exception as e:
        logger.error(f"Error in payment agent: {str(e)}")
        return {
            "output": "I'm experiencing a technical issue with payment processing. Please try again or contact our support team.",
            "stage": "error"
        }

def process_payment(session_id: str, payment_info: Dict[str, str], product: str, plan_name: str) -> Dict[str, Any]:
    """Process the final payment and store user data."""
    try:
        name = payment_info["name"]
        email = payment_info["email"]
        
        # Store user plan data to MongoDB
        # storage_success = store_user_plan(name, email, product, plan_name)  # COMMENTED OUT - Not storing payment details
        
        # if storage_success:  # COMMENTED OUT - Not storing payment details
        #     logger.info(f"Payment processed successfully for {name} - {product} {plan_name}")  # COMMENTED OUT
        # else:  # COMMENTED OUT - Not storing payment details
        #     logger.warning(f"Payment processed but storage failed for {name}")  # COMMENTED OUT
        
        logger.info(f"Payment flow completed for {name} - {product} {plan_name} (details not stored)")
        
        # Generate dummy payment link
        payment_link = f"https://hlas-payment.com/pay/{session_id}?plan={plan_name}&product={product}"
        
        response = f"""*Payment Details Confirmed*

Customer: *{name}*
Email: *{email}*
Product: *{product} Insurance*
Plan: *{plan_name}*

Policy details sent to: {email}

*Complete payment:*
{payment_link}

Policy activates after payment confirmation.

Need help? Contact support anytime."""

        return {
            "output": response,
            "stage": "completed",
            "payment_link": payment_link
        }
        
    except Exception as e:
        logger.error(f"Error processing payment: {str(e)}")
        return {
            "output": "There was an issue processing your payment. Please contact our support team for assistance.",
            "stage": "error"
        }