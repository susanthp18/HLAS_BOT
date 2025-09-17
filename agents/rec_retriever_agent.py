import logging
from app.config import llm
from utils.weaviate_client import get_weaviate_client
from weaviate.classes.query import Filter

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage

def get_available_tiers(product: str) -> list:
    """Returns a list of available plan tiers for a given product."""
    if product.upper() == "TRAVEL":
        return ["Basic", "Silver", "Gold", "Platinum"]
    # elif product.upper() == "MAID":
    #     return ["Basic", "Standard", "Premier"]
    else:
        # Fallback for unknown products
        return ["Basic", "Standard", "Premium"]

class RecRetrieverAgent:

    def get_recommendation_message(self, product: str, plan_tier: str):
        """
        Generates a comprehensive recommendation message using the LLM.
        """
        try:
            client = get_weaviate_client()
            collection = client.collections.get("Insurance_Knowledge_Base")

            # Filter for benefit documents for the specified product
            filters = Filter.all_of([
                Filter.by_property("product_name").equal(product),
                Filter.by_property("doc_type").equal("benefits")
            ])
            
            logger.info(f"🔍 QUERY DEBUG: Searching for product='{product}', doc_type='benefits', plan='{plan_tier}', limit=15")
            logger.info(f"🔍 QUERY DEBUG: Product type: {type(product)}, Product repr: {repr(product)}")
            
            # Also try a broader query to see what's actually in the database
            logger.info(f"🔍 TESTING: Trying broader query to see all Travel-related docs...")
            test_response = collection.query.fetch_objects(
                filters=Filter.by_property("product_name").equal("Travel"),  # Try title case
                limit=5
            )
            logger.info(f"🔍 TEST RESULT: Found {len(test_response.objects)} objects with product_name='Travel'")
            
            # Try another test with TRAVEL uppercase
            test_response2 = collection.query.fetch_objects(
                filters=Filter.by_property("product_name").equal("TRAVEL"),  # Try uppercase
                limit=5
            )
            logger.info(f"🔍 TEST RESULT: Found {len(test_response2.objects)} objects with product_name='TRAVEL'")

            response = collection.query.fetch_objects(
                filters=filters,
                limit=15
            )
            
            logger.info(f"🔍 RESPONSE DEBUG: Response type: {type(response)}")
            logger.info(f"🔍 RESPONSE DEBUG: Response has objects: {hasattr(response, 'objects')}")
            logger.info(f"🔍 RESPONSE DEBUG: Objects count: {len(response.objects) if hasattr(response, 'objects') else 'N/A'}")
            logger.info(f"🔍 RESPONSE DEBUG: Objects type: {type(response.objects) if hasattr(response, 'objects') else 'N/A'}")

            if not response.objects:
                logger.warning(f"⚠️ No benefit documents found for product: {product}")
                retrieved_benefits = f"No specific benefit details found for {product} insurance."
            else:
                logger.info(f"Retrieved {len(response.objects)} benefit chunks for product: {product}")
                
                # Enhanced debugging for Weaviate objects
                logger.info(f"🔍 DETAILED WEAVIATE OBJECT ANALYSIS:")
                
                for i, obj in enumerate(response.objects):
                    chunk_num = i + 1
                    
                    # Log the entire object structure
                    logger.info(f"  === CHUNK {chunk_num} ANALYSIS ===")
                    logger.info(f"    Object type: {type(obj)}")
                    logger.info(f"    Object UUID: {getattr(obj, 'uuid', 'N/A')}")
                    logger.info(f"    Has properties: {hasattr(obj, 'properties')}")
                    
                    if hasattr(obj, 'properties'):
                        props = obj.properties
                        logger.info(f"    Properties type: {type(props)}")
                        logger.info(f"    Properties keys: {list(props.keys()) if props else 'None'}")
                        
                        # Check each expected property
                        content = props.get('content') if props else None
                        product_name = props.get('product_name') if props else None
                        doc_type = props.get('doc_type') if props else None
                        source_file = props.get('source_file') if props else None
                        
                        logger.info(f"    Content exists: {content is not None}")
                        logger.info(f"    Content type: {type(content)}")
                        logger.info(f"    Content length: {len(content) if content else 0}")
                        logger.info(f"    Product name: {product_name}")
                        logger.info(f"    Doc type: {doc_type}")
                        logger.info(f"    Source file: {source_file}")
                        
                        if content:
                            logger.info(f"    Content preview: {repr(content[:100])}...")
                        else:
                            logger.info(f"    Content is: {repr(content)}")
                    else:
                        logger.info(f"    No properties attribute found")
                    
                    # Original logging for compatibility
                    content_for_display = obj.properties.get('content', 'N/A') if hasattr(obj, 'properties') and obj.properties else 'NO_PROPERTIES'
                    logger.info(f"  Chunk {chunk_num}: {content_for_display}")
                    logger.info(f"  ========================")
                
                # Filter and join valid content
                valid_chunks = []
                for obj in response.objects:
                    if hasattr(obj, 'properties') and obj.properties:
                        content = obj.properties.get('content')
                        if content and content.strip():
                            valid_chunks.append(content)
                
                logger.info(f"🔍 SUMMARY: Found {len(valid_chunks)} valid chunks out of {len(response.objects)} total")
                retrieved_benefits = "\n---\n".join(valid_chunks)

        except Exception as e:
            logger.error(f"Error retrieving benefits for {product}: {e}")
            retrieved_benefits = f"I'm having trouble accessing the {product} insurance benefits right now."

        # Get tier names for the product
        tier_names = get_available_tiers(product)
        try:
            current_tier_index = tier_names.index(plan_tier)
            lower_tiers = tier_names[:current_tier_index]
            higher_tiers = tier_names[current_tier_index + 1:]
        except ValueError:
            # If plan_tier not found in list, use defaults
            lower_tiers = []
            higher_tiers = []
        
        # Generate the recommendation message using the LLM
        prompt = [
            SystemMessage(
                content=f"""You are an expert insurance assistant. Your task is to craft a comprehensive recommendation message following this EXACT structure:

REQUIRED FORMAT:
"We recommend the {plan_tier} plan. [Provide a single, concise reason for the recommendation].

[Extract and present ONLY the {plan_tier} plan benefits from all the chunks below]

[Then provide tier alternatives and next steps]"

INPUTS PROVIDED:
- Product: {product}
- Recommended Plan Tier: {plan_tier}
- ALL Product Benefit Details: {retrieved_benefits}
- Available Lower Tiers: {lower_tiers}
- Available Higher Tiers: {higher_tiers}

INSTRUCTIONS:
1. Start with a single, concise sentence explaining why the {plan_tier} is recommended.
2. From the retrieved chunks, extract and present ONLY the benefits specific to the {plan_tier} plan.
3. Include tier alternatives based on available options:
   - If lower tiers available: "If you want lower coverage, you can opt for {', '.join(lower_tiers) if lower_tiers else 'N/A'}"
   - If higher tiers available: "If you want higher coverage, consider {', '.join(higher_tiers) if higher_tiers else 'N/A'}"
4. End with options: additional info, add-ons, or proceed to purchase
5. Use a friendly, professional tone with clear formatting
6. Focus only on the recommended tier's benefits, ignore other tiers' details"""
            ),
            HumanMessage(content="Please generate the recommendation message following the exact format specified."),
        ]
        
        response = llm.invoke(prompt)
        
        return response.content

rec_retriever_agent = RecRetrieverAgent()

def get_recommendation_message(product: str, plan_tier: str):
    """
    Main entry point for generating the recommendation message.
    """
    return rec_retriever_agent.get_recommendation_message(product, plan_tier)

def generate_plan_comparison_table(product: str, requested_tiers: list, user_query: str):
    """
    Fetches all benefit chunks for a product and uses an LLM to generate a markdown comparison table.
    """
    logger.info(f"Generating plan comparison table for {product}, highlighting {requested_tiers}, focusing on '{user_query}'")

    try:
        # 1. Fetch ALL benefit chunks from Weaviate
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")

        response = collection.query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("product_name").equal(product),
                Filter.by_property("doc_type").equal("benefits")
            ]),
            limit=100  # Set a high limit to fetch all benefit chunks
        )

        if not response.objects:
            logger.warning(f"No benefit documents found for product: {product}")
            return "I couldn't find any benefit information to create a comparison. Please check if the documents have been embedded correctly."

        # 2. Combine all chunks into a single context string
        context_str = "\n---\n".join([obj.properties['content'] for obj in response.objects])
        logger.info(f"Combined {len(response.objects)} benefit chunks into a single context of {len(context_str)} characters.")

        # 3. Get available tiers to inform the LLM
        available_tiers = get_available_tiers(product)

        # 4. Create an enhanced prompt for the LLM
        system_prompt = f"""You are an expert insurance assistant. Your task is to create a detailed and accurate comparison of insurance plan tiers, formatted for easy reading on WhatsApp.

PRODUCT: {product} Insurance
ALL AVAILABLE TIERS: {', '.join(available_tiers)}
USER IS INTERESTED IN: {', '.join(requested_tiers) if requested_tiers else 'all tiers'}
USER'S SPECIFIC QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
1.  **Analyze the User's Query First**: Before looking at the context, break down the user's query to identify the specific benefit, plan, and any special conditions (like 'child', 'COVID-19', or 'pre-existing conditions').
2.  **Match Context Precisely**: Scan the provided 'BENEFITS INFORMATION' for sections that exactly match the user's query. For example, if the user asks about "Overseas Medical Expenses", you must differentiate between the general benefit and more specific ones like "Overseas Medical Expenses due to COVID-19" or "Add-On... Pre-Existing Condition Overseas Medical Expenses".
3.  **Prioritize Specificity**: Only use information from a specific section (like COVID-19 or Pre-Existing) if the user's query contains those exact keywords. Otherwise, you MUST use the general benefit information.
4.  **Handle Ambiguity**: If the user's query is slightly ambiguous and could match multiple benefit sections (e.g., a general benefit and a COVID-19 specific one), you MUST present the information for **both** in your comparison, clearly labeling each one (e.g., "• General Medical Expenses: $X", "• COVID-19 Medical Expenses: $Y").
5.  **Create a Focused Comparison**: For each of the tiers the user is interested in, create a separate section starting with the plan name in bold (e.g., *Gold Plan*).
6.  **List Relevant Benefits**: Under each plan name, list the key benefits using bullet points (•). If the user asked about a specific benefit, that MUST be the first bullet point.
7.  **Add a Final Summary**: After the details, add a "SUMMARY" section. This should be a short, 2-3 sentence comparison of the key differences, focusing on the topic of the user's specific question.
8.  **Accuracy is Key**: Ensure all information is accurate and based *only* on the provided context. Do not use markdown tables.
"""

        human_prompt = f"""BENEFITS INFORMATION:
---
{context_str}
---

Now, please generate the WhatsApp-friendly comparison and summary based on the critical instructions.
"""
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # 5. Call the LLM to generate the table
        response = llm.invoke(prompt)
        logger.info("Successfully generated plan comparison text.")
        return response.content

    except Exception as e:
        logger.error(f"Failed to generate plan comparison table: {e}")
        return "I'm sorry, I encountered an error while trying to create the plan comparison. Please try again later."