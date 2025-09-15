import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient, errors

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "hlas_bot"
COLLECTION_NAME = "conversations"

def initialize_mongo():
    """
    Connects to MongoDB, creates the database and collection if they don't exist,
    and sets up the necessary indexes for the conversation history.
    """
    print("--- MongoDB Initialization Script ---")
    
    try:
        # Connect to MongoDB
        print(f"Connecting to MongoDB at {MONGO_URI}...")
        client = MongoClient(MONGO_URI)
        
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("✅ MongoDB connection successful.")
        
        # Get database and collection
        db = client[DB_NAME]
        
        # Check if collection exists, create if not
        if COLLECTION_NAME not in db.list_collection_names():
            print(f"Collection '{COLLECTION_NAME}' does not exist. Creating it now...")
            db.create_collection(COLLECTION_NAME)
            print(f"✅ Collection '{COLLECTION_NAME}' created in database '{DB_NAME}'.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists in database '{DB_NAME}'.")
            
        collection = db[COLLECTION_NAME]
        
        # --- Index Creation ---
        print("\nCreating indexes...")
        
        # 1. Create a unique index on session_id to prevent duplicates
        try:
            collection.create_index("session_id", unique=True)
            print("  - ✅ Unique index on 'session_id' ensured.")
        except errors.OperationFailure as e:
            if "already exists" in str(e):
                print("  - ℹ️ Unique index on 'session_id' already exists.")
            else:
                raise e
                
        # 2. Create an index on last_active for efficient cleanup of old sessions
        try:
            collection.create_index("last_active")
            print("  - ✅ Index on 'last_active' ensured.")
        except errors.OperationFailure as e:
            if "already exists" in str(e):
                print("  - ℹ️ Index on 'last_active' already exists.")
            else:
                raise e
        
        print("\n✅ MongoDB initialization complete.")

    except errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("   Please ensure that your MongoDB server is running and that the MONGO_URI in your .env file is correct.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()
            print("\nMongoDB connection closed.")

if __name__ == "__main__":
    initialize_mongo()
