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
        system_prompt = f"""You are an expert insurance assistant. Your primary function is to create an accurate comparison, based ONLY on the provided context. You must act as a filter, discarding any information that is not a perfect match for the user's query.

USER'S SPECIFIC QUESTION: {user_query}
USER IS INTERESTED IN: {', '.join(requested_tiers) if requested_tiers else 'all tiers'}

GENERAL RULES:
- Normalize plan spelling: if the context uses 'Platinium', label it as 'Platinum' in the output.
- Do NOT include specialized sections (COVID-19, Pre-Existing Conditions/Add-Ons) unless the user's question explicitly mentions them.
- Preserve exact vocabulary for amounts: 'Unlimited', 'Not available', currency symbols, and caps (e.g., 'capped at S$100 per visit').
- Age categories appear in the context and must be respected when relevant:
  - Adult (age 70 years and below)
  - Adult (age above 70 years)
  - Child
- **Strict Length Limit:** The entire response MUST be under 3500 characters to comply with API limits. Be concise.

**YOUR TASK - A STRICT 3-STEP PROCESS:**

**STEP 1: IDENTIFY THE EXACT BENEFIT(S)**
- Analyze the user's question. If it's a general comparison (no specific benefit mentioned), you MUST compare ALL general benefits found in the context for the requested tiers.
- If the question targets a specific benefit (e.g., 'Overseas Medical Expenses'), focus only on that benefit family and DO NOT list other benefits.
- **RULE OF SPECIFICITY**: Use the general benefit sections unless the user's question explicitly mentions a specialized section.
- **DISAMBIGUATE 'MEDICAL EXPENSES'**: If the user says 'Medical Expenses' without qualifiers, include BOTH 'Overseas Medical Expenses' and 'Medical Expenses in Singapore' (with the S$100 per-visit cap when applicable).

 BENEFIT FILTERING LOGIC (CRITICAL):
 - Derive an ALLOWED_BENEFITS whitelist from the user's question using exact or synonymous phrases:
   - 'emergency medical evacuation', 'evacuation', 'med evac' → Emergency Medical Evacuation
   - 'medical expenses' (generic) → Overseas Medical Expenses AND Medical Expenses in Singapore
   - 'overseas medical expenses' → Overseas Medical Expenses
   - 'medical expenses in singapore', 'singapore medical expenses' → Medical Expenses in Singapore
   - 'repatriation' → Repatriation of Mortal Remains
 - If one or more specific benefits are implied, ALLOWED_BENEFITS must contain ONLY those families; compare ONLY those.
 - If the question mentions no benefit, set ALLOWED_BENEFITS = ALL and compare all general benefits.

**STEP 2: EXTRACT THE DATA**
- Restrict output to ONLY the requested tiers (if none specified, include all tiers).
- For each benefit you include, extract the exact amounts for the requested tiers from the context. If a value is not present for a tier, state 'Not available'.
- For age-based benefits, show values per age group (Adult ≤70, Adult >70, Child). If age not specified by user, present all age categories that exist in context.

**STEP 3: FORMULATE THE COMPARISON**
- Create a clear, WhatsApp-friendly comparison.
- If the user asked for a general comparison, structure by CATEGORIES and then benefits:
  1) **Medical And Other Expenses** (include all listed sub-benefits such as Overseas Medical Expenses, Medical Expenses in Singapore, Compassionate Visit, Repatriation of Mortal Remains, Overseas Funeral Expenses, Return of Minor Children, Emergency Medical Evacuation, Overseas Hospital Cash Benefit, Hospital Cash Benefit in Singapore)
  2) **Personal Accident**
  3) **Travel Inconvenience**
  4) **Liability**
  5) **Lifestyle**
- If the user asked for a specific benefit, limit output strictly to that benefit family.
- Under each benefit, list ONLY the requested tiers with amounts (by age group when relevant). Do NOT omit benefits when the query is general.
 - HARD CONSTRAINT: If ALLOWED_BENEFITS is not ALL, DO NOT include any benefit outside that whitelist. Do not add extra sections.
- Add a short SUMMARY at the end highlighting key differences tied to the user's question.
- Keep formatting concise with bold headings and bullet points. Avoid tables.
"""

        human_prompt = f"""BENEFITS INFORMATION:
---
{context_str}
---

FINAL OUTPUT INSTRUCTION:
- If the user did NOT specify a particular benefit, you MUST provide an exhaustive comparison covering ALL general benefits for the requested tiers, organized by the category order above. Do NOT skip any benefit present in the context.
- If the user specified a particular benefit (e.g., 'Overseas Medical Expenses'), output ONLY that benefit family and nothing else.
- Output ONLY the final comparison and summary; do NOT include any intermediate reasoning.

USER QUESTION (use this to build ALLOWED_BENEFITS):
{user_query}

Now, execute this 3-step process and provide the final comparison and summary.
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


def generate_plan_summary(product: str, plan_tier: str):
    """
    Generates a comprehensive, WhatsApp-friendly summary of a SINGLE plan tier
    across ALL general benefits (excluding COVID-19 and Pre-Existing Add-Ons unless explicitly asked).
    """
    logger.info(f"Generating plan summary for {product} - tier '{plan_tier}'")

    try:
        # Fetch ALL benefit chunks for the product
        client = get_weaviate_client()
        collection = client.collections.get("Insurance_Knowledge_Base")

        response = collection.query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("product_name").equal(product),
                Filter.by_property("doc_type").equal("benefits")
            ]),
            limit=100
        )

        if not response.objects:
            logger.warning(f"No benefit documents found for product: {product}")
            return f"I couldn't find any benefit information for the {product} {plan_tier} plan. Please check if the documents have been embedded correctly."

        # Combine chunks into a single context string
        context_str = "\n---\n".join([obj.properties['content'] for obj in response.objects])
        logger.info(f"Combined {len(response.objects)} benefit chunks into a single context for summary.")

        # Normalize plan tier label (handle common misspelling)
        normalized_tier = "Platinum" if plan_tier.strip().lower() in ["platinum", "platinium"] else plan_tier.strip().title()

        system_prompt = f"""You are an expert insurance assistant. Your task is to produce a complete, accurate summary of a SINGLE plan tier's benefits using ONLY the provided context.

GENERAL RULES:
- Plan tier to summarize: {normalized_tier}
- Normalize spelling in output: use 'Platinum' (not 'Platinium').
- Exclude specialized sections (COVID-19, Pre-Existing/Add-Ons) unless explicitly requested (not requested here).
- Preserve exact vocabulary for amounts: 'Unlimited', 'Not available', currency symbols, and caps (e.g., 'capped at S$100 per visit').
- Respect age categories where present: Adult (≤70), Adult (>70), Child. If multiple categories exist, present them clearly.

STRICT PROCESS:
1) Identify ONLY the general benefit sections applicable to all travelers (not COVID/add-ons).
2) Extract the exact amounts for the {normalized_tier} plan for each benefit.
3) Produce a WhatsApp-friendly summary with bold section headings and bullet points. Avoid tables.

STRUCTURE:
- **Medical And Other Expenses**
  - Include sub-benefits: Overseas Medical Expenses, Medical Expenses in Singapore (note S$100 per-visit cap where applicable), Compassionate Visit, Repatriation of Mortal Remains, Overseas Funeral Expenses, Return of Minor Children, Emergency Medical Evacuation, Overseas Hospital Cash Benefit, Hospital Cash Benefit in Singapore
- **Personal Accident**
- **Travel Inconvenience**
- **Liability**
- **Lifestyle**

OUTPUT:
- Show values for the {normalized_tier} plan only.
- For age-based items, list the categories with amounts.
- End with a short 2–3 sentence summary.
"""

        human_prompt = f"""BENEFITS INFORMATION:
---
{context_str}
---

FINAL OUTPUT INSTRUCTION:
- Output ONLY the formatted summary for the {normalized_tier} plan and the short summary; do NOT include any intermediate reasoning.
"""

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        result = llm.invoke(prompt)
        logger.info("Successfully generated single plan summary text.")
        return result.content

    except Exception as e:
        logger.error(f"Failed to generate plan summary: {e}")
        return "I'm sorry, I encountered an error while creating the plan summary. Please try again later."