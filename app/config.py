# app/config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Chat History
MAX_CONTEXT_MESSAGES = 10

from utils.llm_services import llm, embedding_model