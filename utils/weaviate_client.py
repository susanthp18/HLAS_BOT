import os
import logging
import weaviate
from typing import Optional
from urllib.parse import urlparse
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter

logger = logging.getLogger(__name__)

# Global Weaviate client instance
_weaviate_client = None

def get_weaviate_client():
    """
    Get a singleton Weaviate client instance.
    """
    global _weaviate_client
    if _weaviate_client is None:
        try:
            parsed_url = urlparse(os.getenv("WEAVIATE_URL"))
            auth_credentials = None
            if os.getenv("WEAVIATE_API_KEY"):
                auth_credentials = AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
            
            _weaviate_client = weaviate.connect_to_custom(
                http_host=parsed_url.hostname,
                http_port=parsed_url.port,
                http_secure=parsed_url.scheme == "https",
                grpc_host=parsed_url.hostname,
                grpc_port=50051,
                grpc_secure=False,
                auth_credentials=auth_credentials,
            )
            logger.info("Successfully connected to Weaviate.")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    return _weaviate_client

