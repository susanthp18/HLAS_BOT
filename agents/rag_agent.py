import logging
from app.config import llm
from utils.weaviate_client import get_weaviate_client
from weaviate.classes.query import Filter, TargetVectors
from langchain_core.messages import SystemMessage, HumanMessage
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt for the RAG agent
RAG_SYSTEM_PROMPT = """
You are an expert AI assistant for HLAS, specializing in insurance products. Your task is to answer user questions based *only* on the provided context.

**Strict 3-Step Process:**
1.  **Identify Exact Benefit:** Meticulously scan the provided context to find the section that *exactly* matches the user's query. Pay very close attention to headings like "General Benefits," "COVID-19 Benefits," or "Add-On Coverages."
2.  **Extract Raw Data:** Once the correct section is identified, extract the specific data points (e.g., plan names, coverage amounts) from that section *only*.
3.  **Formulate Answer:** Use the extracted raw data to construct a clear, concise, and accurate answer to the user's question.

**CRITICAL RULES:**
*   **Rule of Specificity:** You MUST differentiate between general benefits and specific, conditional benefits (e.g., for COVID-19 or pre-existing conditions). If the user asks a general question like "What is the coverage for medical expenses?", you MUST use the general benefits section and NOT the COVID-19 or other specialized sections.
*   **Handle Ambiguity:** If the user's query is slightly ambiguous and could plausibly match two sections (e.g., a general benefit and a COVID-19 specific one), present the information for **BOTH** sections and clearly label them. For example: "For general overseas medical expenses, the coverage is X. For COVID-19 related overseas medical expenses, the coverage is Y."
*   **No Outside Information:** Do not use any knowledge outside of the provided context.
*   **If Not Found:** If the information is not in the context, state that you cannot find the details and offer to connect the user to a live agent.
*   **Format for Clarity:** Use bullet points and bold text to make the information easy to read on WhatsApp.
*   **Strict Length Limit:** The entire response MUST be under 3500 characters to comply with API limits.
"""

RAG_DOMAIN_RULES = """
Domain Rules for Travel Benefits in the Provided Context:

- Tiers: Basic, Silver, Gold, Platinum. If the user mentions specific tiers (e.g., "basic and gold"), restrict the answer to ONLY those tiers.
- Age Groups: Some benefits vary by age. Categories used in the context are:
  - Adult (age 70 years and below)
  - Adult (age above 70 years)
  - Child
  If the user does not specify age, present values for all relevant age categories clearly labeled, or default to Adult (≤70) AND add a brief note for the other categories when present.
- Medical Expenses is a family of benefits under "Medical And Other Expenses":
  - Overseas Medical Expenses (age-based amounts)
  - Medical Expenses in Singapore (age-based amounts, capped at S$100 per visit where no initial treatment was sought overseas)
  If the user asks generally for "Medical Expenses" without saying "Overseas" or "in Singapore", present BOTH sections, clearly labeled.
- Emergency Medical Evacuation (general) under "Medical And Other Expenses" is Unlimited across all tiers. Do NOT confuse this with COVID-19 or Pre-Existing Conditions sections.
- Special Sections (include ONLY if explicitly asked):
  - Enhanced Medical Benefits for COVID-19
  - Add-On Coverages for Pre-Existing Medical Conditions
  If the user’s question does not include the keywords (e.g., "COVID", "pre-existing"), do not use those sections.
- Vocabulary: Use exact terms from the context like "Unlimited" and "Not available". Preserve S$ caps where stated (e.g., "capped at S$100 per visit").
- Output Discipline:
  - Use bold section labels (e.g., **Medical Expenses (Overseas)**, **Medical Expenses in Singapore**).
  - Under each, list only the requested tiers with amounts per age group.
  - Keep it concise and WhatsApp-friendly. Avoid tables.
"""

class RAGAgent:
    def __init__(self):
        pass

    def answer_query(self, query: str, chat_history: list, product: str = None):
        """
        Answers a user's query using a native Weaviate hybrid search.
        """
        if not product:
            # return "Please specify which insurance product you're asking about (Travel or Maid)."
            return "Please specify that you are asking about Travel insurance."

        try:
            logger.info(f"🔍 RAG QUERY - Product: {product}, Query: '{query}'")
            
            client = get_weaviate_client()
            collection = client.collections.get("Insurance_Knowledge_Base")

            # Generate embedding for the user query using Azure OpenAI
            from utils.llm_services import embedding_model
            logger.info(f"🔍 Generating query embedding for: '{query}'")
            query_embedding = embedding_model.embed_query(query)
            
            # Perform hybrid search with NAMED VECTORS
            logger.info(f"🔍 Performing hybrid search with named vectors + average join strategy")
            response = collection.query.hybrid(
                query=query,                           # For BM25 keyword matching
                vector={
                    "content_vector": query_embedding,     # Semantic match with content
                    "questions_vector": query_embedding    # Semantic match with questions  
                },
                target_vector=TargetVectors.average(["content_vector", "questions_vector"]),  # Average join strategy
                limit=15,
                alpha=0.7,  # 70% semantic + 30% keyword (more conceptual)
                filters=Filter.by_property("product_name").equal(product),
                query_properties=["content", "questions"]  # BM25 searches both
            )

            if not response.objects:
                logger.warning(f"⚠️ No documents found for query: '{query}'")
                return "I couldn't find any specific information for your query. Could you please rephrase it?"

            # --- DEBUG LOGGING: Print retrieved chunks ---
            logger.info(f"📚 Retrieved {len(response.objects)} chunks for RAG synthesis.")
            
            # Detailed object analysis for debugging
            for i, obj in enumerate(response.objects):
                chunk_num = i + 1
                logger.info(f"  === CHUNK {chunk_num} ANALYSIS ===")
                logger.info(f"    Object UUID: {getattr(obj, 'uuid', 'N/A')}")
                
                if hasattr(obj, 'properties'):
                    props = obj.properties
                    logger.info(f"    Properties keys: {list(props.keys()) if props else 'None'}")
                    
                    # Log each property for detailed inspection
                    for key, value in props.items():
                        value_preview = repr(value)[:100] + '...' if isinstance(value, str) and len(value) > 100 else repr(value)
                        logger.info(f"      - {key}: {value_preview}")
                else:
                    logger.info(f"    No properties attribute found")
                logger.info(f"  ========================")

            # --- END DEBUG LOGGING ---

            # Synthesize the final answer using the retrieved chunks
            context_str = "\n---\n".join([obj.properties['content'] for obj in response.objects])
            
            prompt = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                SystemMessage(content=RAG_DOMAIN_RULES),
                HumanMessage(content=(
                    f"CONTEXT:\n\n{context_str}\n\n"
                    f"USER QUESTION: {query}\n\n"
                    "FINAL OUTPUT INSTRUCTION: Provide ONLY the final answer. Do not show intermediate steps or reasoning."
                ))
            ]

            llm_response = llm.invoke(prompt)
            formatted_response = self._add_guidance(llm_response.content, product, query)

            logger.info(f"✅ RAG Response Generated: {len(formatted_response)} characters")
            return formatted_response

        except Exception as e:
            logger.error(f"❌ Error querying {product} collection: {e}")
            return f"I'm having trouble accessing the {product} insurance information right now. Could you please rephrase your question?"


    def _add_guidance(self, response_text: str, product: str, original_query: str):
        """
        Adds contextual guidance to the response.
        """
        # Check if this is a plan comparison query (called from recommendation stage)
        is_plan_comparison = "plan tiers" in original_query.lower() or "comparison" in original_query.lower()
        
        # Only add guidance for non-plan-comparison queries to avoid duplication
        if not is_plan_comparison:
            # Analyze if user might be ready for purchase (based on query type)
            purchase_keywords = ["price", "cost", "buy", "purchase", "how much", "payment", "premium"]
            is_purchase_related = any(keyword in original_query.lower() for keyword in purchase_keywords)
            
            if is_purchase_related:
                # Add purchase guidance for purchase-related queries
                response_text += f"\n\n*Ready to Purchase?*\nType 'proceed with purchase' to continue."
            else:
                # Add standard guidance for general queries
                response_text += f"\n\nNeed more details? Just ask.\nReady to purchase? Type 'buy {product.lower()} insurance'."
        
        return response_text

rag_agent = RAGAgent()

def get_rag_response(query: str, chat_history: list, product: str = None):
    """
    Main entry point for answering a user's query.
    """
    return rag_agent.answer_query(query, chat_history, product)

# Legacy function removed - use get_rag_response() directly