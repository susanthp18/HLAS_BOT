import logging
import json
from app.session_manager import get_collected_info
from utils.llm_services import llm
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RecommendedPlan(BaseModel):
    """Pydantic model for the recommended insurance plan."""
    plan: str = Field(..., description="The recommended plan tier (e.g., 'Gold', 'Silver').")
    user_has_specific_requirements: bool = Field(..., description="Set to true if the user has specific needs.")
    confidence: float = Field(..., description="Confidence score for the recommendation (0.0 to 1.0).")
    recommendation_reason: str = Field(..., description="The reason for recommending this plan.")
    next_step_prompt: str = Field(..., description="A prompt guiding the user to the next step.")

class RecommendationAgent:
    def recommend_plan(self, session_id: str, product: str):
        """
        Recommends a plan based on the collected information for a given product using an LLM.
        """
        logger.info(f"🔍 RECOMMENDATION DEBUG: Starting recommendation for session {session_id}, product {product}")
        
        collected_info = get_collected_info(session_id)
        logger.info(f"🔍 RECOMMENDATION DEBUG: Collected info: {collected_info}")
        
        if not collected_info or 'travel_info' not in collected_info:
            logger.error(f"🔍 RECOMMENDATION DEBUG: No travel_info found for session {session_id}")
            return {"plan": "not available", "reason": "No user information available for recommendation"}
        
        travel_info = collected_info['travel_info']
        budget_preference = travel_info.get('budget_preference', '').lower()

        # Determine the plan based on budget preference
        if budget_preference == 'budget-friendly':
            recommended_plan = 'Silver'
            plan_context = "The user chose a 'budget-friendly' option. Briefly explain why the Silver plan is a good balance of cost and coverage. Also, mention that the 'Basic' plan is available for essential-only coverage, and 'Gold' or 'Platinum' plans are available for more comprehensive benefits."
        elif budget_preference == 'comprehensive':
            recommended_plan = 'Gold'
            plan_context = "The user chose a 'comprehensive' option. Briefly explain why the Gold plan offers extensive coverage for their needs. Also, mention that the 'Platinum' plan is our top-tier option for maximum protection, and that 'Silver' is available if they'd like to see a more budget-conscious choice."
        else:
            # Fallback if budget_preference is missing or invalid
            recommended_plan = 'Silver'
            plan_context = "Since a budget preference wasn't specified, the Silver plan is a good starting point as it offers a great balance of features and cost. Mention that other plans like 'Basic', 'Gold', and 'Platinum' are also available."

        # Use a structured output chain to ensure JSON format
        chain = llm.with_structured_output(RecommendedPlan, method="function_calling")

        # Define the system prompt with clear instructions for generating the text
        system_prompt = f"""You are an expert insurance recommendation agent for HLAS. Your goal is to generate the user-facing text for a pre-determined insurance plan recommendation.

TASK:
- The plan has already been chosen for the user. Your job is to create the `recommendation_reason` and `next_step_prompt` for the '{recommended_plan}' plan.
- Use the user's information and the provided plan context to generate helpful and personalized text.
- Do NOT change the plan.
"""

        # Define the human prompt with the user's data and the chosen plan
        human_prompt = f"""User's information:
{json.dumps(travel_info, indent=2)}

Chosen Plan: {recommended_plan}

Context for your response: {plan_context}

Please generate the `recommendation_reason` and `next_step_prompt` in the required JSON format."""

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
        
        logger.info(f"🔍 RECOMMENDATION DEBUG: Calling LLM with structured output chain.")
        try:
            response = chain.invoke(prompt)
            # The response is now a Pydantic object
            recommendation = response.model_dump()
            # Manually add the pre-determined plan to the final dict
            recommendation['plan'] = recommended_plan
            logger.info(f"🔍 RECOMMENDATION DEBUG: Successfully got structured response: {recommendation}")
            
        except Exception as e:
            logger.error(f"🔍 RECOMMENDATION DEBUG: LLM structured output call failed: {str(e)}")
            # Fallback mechanism
            recommendation = {"plan": recommended_plan, "reason": f"LLM call failed: {str(e)}"}
            
        return recommendation

recommendation_agent = RecommendationAgent()

def get_recommendation(session_id: str, product: str):
    """
    Main entry point for getting a plan recommendation.
    """
    return recommendation_agent.recommend_plan(session_id, product)