import os
import logging
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Global instances for the models
_llm_instance = None
_embedding_model_instance = None

def get_llm():
    """
    Get a singleton instance of the AzureChatOpenAI LLM.
    """
    global _llm_instance
    if _llm_instance is None:
        try:
            _llm_instance = AzureChatOpenAI(
                temperature=0,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            )
            logger.info("Successfully initialized Azure OpenAI LLM.")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI LLM: {e}")
            raise
    return _llm_instance

def get_embedding_model():
    """
    Get a singleton instance of the AzureOpenAIEmbeddings model (LangChain).
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        try:
            _embedding_model_instance = AzureOpenAIEmbeddings(
                model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
            logger.info("Successfully initialized Azure OpenAI Embedding model (LangChain).")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI Embedding model: {e}")
            raise
    return _embedding_model_instance

# Initialize the models at startup
llm = get_llm()
embedding_model = get_embedding_model()