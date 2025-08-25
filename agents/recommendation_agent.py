import logging
import json
from app.session_manager import get_collected_info
from utils.llm_services import llm
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class RecommendationAgent:
    def recommend_plan(self, session_id: str, product: str):
        """
        Recommends a plan based on the collected information for a given product using an LLM.
        """
        logger.info(f"🔍 RECOMMENDATION DEBUG: Starting recommendation for session {session_id}, product {product}")
        
        collected_info = get_collected_info(session_id)
        logger.info(f"🔍 RECOMMENDATION DEBUG: Collected info: {collected_info}")
        
        if not collected_info:
            logger.error(f"🔍 RECOMMENDATION DEBUG: No collected info found for session {session_id}")
            return {"plan": "not available", "reason": "No user information available for recommendation"}
        
        # Use the LLM to generate a recommendation in JSON format
        prompt = [
            SystemMessage(
                content=f"""You are an expert insurance recommendation assistant. Your task is to recommend an insurance plan for the following product: {product}.

Analyze the user's information and recommend a plan tier (e.g., Basic, Silver, Gold, Platinum for Travel; Basic, Enhanced, Premier, Exclusive for Maid).

User information: {collected_info}

Respond with a JSON object containing a single key: "plan".
"""
            ),
            HumanMessage(content="Please recommend a plan in JSON format."),
        ]
        
        logger.info(f"🔍 RECOMMENDATION DEBUG: Calling LLM with prompt")
        try:
            response = llm.invoke(prompt)
            logger.info(f"🔍 RECOMMENDATION DEBUG: LLM response: {response.content}")
        except Exception as e:
            logger.error(f"🔍 RECOMMENDATION DEBUG: LLM call failed: {str(e)}")
            return {"plan": "not available", "reason": f"LLM call failed: {str(e)}"}
        
        try:
            # Clean the response content to remove markdown formatting
            cleaned_content = response.content.strip().replace('```json', '').replace('```', '').strip()
            recommendation = json.loads(cleaned_content)
            logger.info(f"🔍 RECOMMENDATION DEBUG: Successfully parsed JSON: {recommendation}")
        except json.JSONDecodeError as e:
            logger.error(f"🔍 RECOMMENDATION DEBUG: Failed to decode LLM response into JSON: {str(e)}")
            logger.error(f"🔍 RECOMMENDATION DEBUG: Raw response content: {response.content}")
            # Fallback mechanism
            recommendation = {"plan": "not available"}
            
        return recommendation

recommendation_agent = RecommendationAgent()

def get_recommendation(session_id: str, product: str):
    """
    Main entry point for getting a plan recommendation.
    """
    return recommendation_agent.recommend_plan(session_id, product)