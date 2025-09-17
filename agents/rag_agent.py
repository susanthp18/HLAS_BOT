import logging
from app.config import llm
from utils.weaviate_client import get_weaviate_client
from weaviate.classes.query import Filter, TargetVectors
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class RAGAgent:
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
                limit=5,
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
                SystemMessage(content=f"""You are a professional insurance assistant. Your task is to answer the user's query with extreme precision based ONLY on the provided context.

CRITICAL INSTRUCTIONS:
1.  **Analyze the User's Query First**: Before looking at the context, break down the user's query to identify the specific benefit, plan, and any special conditions (like 'child', 'COVID-19', or 'pre-existing conditions').
2.  **Match Context Precisely**: Scan the provided CONTEXT for sections that exactly match the user's query. For example, if the user asks about "Overseas Medical Expenses", you must differentiate between the general benefit and more specific ones like "Overseas Medical Expenses due to COVID-19" or "Add-On... Pre-Existing Condition Overseas Medical Expenses".
3.  **Prioritize Specificity**: Only use information from a specific section (like COVID-19 or Pre-Existing) if the user's query contains those exact keywords. Otherwise, you MUST use the general benefit information.
4.  **Handle Ambiguity**: If the user's query is slightly ambiguous and could match multiple benefit sections (e.g., a general benefit and a COVID-19 specific one), you MUST present the information for **both**, clearly labeling each one. For example: "For general medical expenses, the coverage is X. For medical expenses related to COVID-19, the coverage is Y."
5.  **Synthesize Carefully**: Formulate your answer using only the information from the precisely matched context. Do not mix information from different benefits.
6.  **WhatsApp Formatting**: Format the response for easy reading on WhatsApp using short bullet points (•) and bolding (*text*) for key terms.

"""),
                HumanMessage(content=f"CONTEXT:\n{context_str}\n\nQUERY:\n{query}\n\nBased on the context and my critical instructions, please provide a detailed and accurate answer to the query.")
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