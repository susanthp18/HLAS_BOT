import uvicorn
import os
import sys
from dotenv import load_dotenv
from uvicorn.config import LOGGING_CONFIG

import warnings
from pydantic import PydanticDeprecatedSince20

# Suppress the specific Pydantic deprecation warning from LlamaIndex
warnings.filterwarnings("ignore")

# Load environment variables from .env file at the very beginning
load_dotenv()

# Add the project root to the system path
# This allows the script to be run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(client_addr)s - "%(request_line)s" %(status_code)s'
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, log_config=LOGGING_CONFIG)