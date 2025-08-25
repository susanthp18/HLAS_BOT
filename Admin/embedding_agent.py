import os
import sys

# Add the project root to the Python path before other imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import argparse
from dotenv import load_dotenv
import re
import json
import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from utils.llm_services import llm, embedding_model

# Define project root and source_db path for robust execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DB_PATH = os.path.join(PROJECT_ROOT, "Admin", "source_db")
DEBUG_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "Admin", "debug_chunks")

# Load environment variables
load_dotenv()

# Setup logging
log_directory = "Admin/logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "embedding_agent.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_all_products():
    """Gets all unique product names by scanning filenames in the source_db directory."""
    products = set()
    if not os.path.isdir(SOURCE_DB_PATH):
        logger.error(f"Source directory not found: {SOURCE_DB_PATH}")
        return []
        
    for doc_type_folder in os.listdir(SOURCE_DB_PATH):
        type_dir_path = os.path.join(SOURCE_DB_PATH, doc_type_folder)
        if os.path.isdir(type_dir_path):
            # Special handling for PDFs where product is a sub-folder
            if doc_type_folder.lower() == 'pdfs':
                for product_folder in os.listdir(type_dir_path):
                    product_path = os.path.join(type_dir_path, product_folder)
                    if os.path.isdir(product_path):
                        products.add(product_folder)
            else:
                # For other types, extract product from filename like "Travel_benefits.txt"
                for filename in os.listdir(type_dir_path):
                    if '_' in filename:
                        product_name = filename.split('_')[0]
                        products.add(product_name)
    logger.info(f"Discovered products: {list(products)}")
    return list(products)

def chunk_benefits(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Chunks a text file with overlap."""
    logger.info(f"Chunking benefits file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)}")
    return chunks

def chunk_faqs(file_path: str):
    """Chunks a FAQ file where each Q&A pair is a chunk."""
    logger.info(f"Chunking FAQ file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split by "Q:" but keep the delimiter
    qa_pairs = re.split(r'(?=^Q:)', text, flags=re.MULTILINE)
    chunks = [pair.strip() for pair in qa_pairs if pair.strip()]
    logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)}")
    return chunks

def chunk_policy_md(file_path: str):
    """Chunks a markdown file by sections (headings)."""
    logger.info(f"Chunking policy file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by lines that start with #, ##, etc. Keep the delimiter.
    chunks = re.split(r'(?=^#+\s)', text, flags=re.MULTILINE)
    
    # Filter out any empty strings that might result from the split
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
    logger.info(f"Created {len(chunks)} chunks from {os.path.basename(file_path)}")
    return chunks

def generate_hypothetical_questions(chunk: str):
    """Generates 10 hypothetical questions for a given text chunk using an LLM."""
    logger.info(f"Generating questions for chunk starting with: '{chunk[:80]}...'")
    
    prompt = [
        SystemMessage(content="""You are an expert at generating hypothetical questions from a given text chunk.
Your task is to generate 10 unique questions that a user might ask, which could be answered by the provided text.
The questions should be varied and cover different aspects of the text.
Return the questions as a JSON object with a single key "questions" which is a list of 10 strings.
Example: {"questions": ["What is the coverage for...?", "How do I claim for...?", ...]}"""),
        HumanMessage(content=f"Here is the text chunk:\n\n---\n{chunk}\n---")
    ]
    
    try:
        response = llm.invoke(prompt)
        # Extract JSON from the response content
        json_str = response.content.strip().replace("```json", "").replace("```", "").strip()
        questions_obj = json.loads(json_str)
        questions = questions_obj.get("questions", [])
        if not isinstance(questions, list) or len(questions) == 0:
            logger.warning("LLM returned empty or invalid question list.")
            return []
        logger.info(f"Successfully generated {len(questions)} questions.")
        return questions
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}\nResponse: {response.content}")
        return []
    except Exception as e:
        logger.error(f"An error occurred during question generation: {e}")
        return []


def save_chunks_to_debug_folder(product_name, all_objects):
    """
    Save chunks and metadata to debug folder for analysis.
    """
    # Create debug folder structure
    product_debug_path = os.path.join(DEBUG_OUTPUT_PATH, product_name)
    os.makedirs(product_debug_path, exist_ok=True)
    
    logger.info(f"Saving {len(all_objects)} chunks to debug folder: {product_debug_path}")
    
    # Save detailed chunk analysis
    chunk_analysis = {
        "product_name": product_name,
        "total_chunks": len(all_objects),
        "chunks_by_doc_type": {},
        "chunks_by_source": {},
        "empty_chunks": [],
        "valid_chunks": [],
        "chunk_details": []
    }
    
    for i, obj in enumerate(all_objects):
        chunk_id = f"{product_name}_{obj['doc_type']}_{i+1}"
        content = obj.get('content', '')
        
        # Analyze chunk
        is_empty = not content or not content.strip()
        content_length = len(content) if content else 0
        
        chunk_detail = {
            "chunk_id": chunk_id,
            "chunk_index": i + 1,
            "doc_type": obj.get('doc_type', 'unknown'),
            "source_file": obj.get('source_file', 'unknown'),
            "content_length": content_length,
            "is_empty": is_empty,
            "has_questions": bool(obj.get('questions', [])),
            "questions_count": len(obj.get('questions', [])),
            "content_preview": content[:200] + "..." if content and len(content) > 200 else content
        }
        
        chunk_analysis["chunk_details"].append(chunk_detail)
        
        # Track by doc_type
        doc_type = obj.get('doc_type', 'unknown')
        if doc_type not in chunk_analysis["chunks_by_doc_type"]:
            chunk_analysis["chunks_by_doc_type"][doc_type] = {"total": 0, "empty": 0, "valid": 0}
        chunk_analysis["chunks_by_doc_type"][doc_type]["total"] += 1
        
        # Track by source file
        source_file = obj.get('source_file', 'unknown')
        if source_file not in chunk_analysis["chunks_by_source"]:
            chunk_analysis["chunks_by_source"][source_file] = {"total": 0, "empty": 0, "valid": 0}
        chunk_analysis["chunks_by_source"][source_file]["total"] += 1
        
        if is_empty:
            chunk_analysis["empty_chunks"].append(chunk_id)
            chunk_analysis["chunks_by_doc_type"][doc_type]["empty"] += 1
            chunk_analysis["chunks_by_source"][source_file]["empty"] += 1
            logger.warning(f"Empty chunk detected: {chunk_id}")
        else:
            chunk_analysis["valid_chunks"].append(chunk_id)
            chunk_analysis["chunks_by_doc_type"][doc_type]["valid"] += 1
            chunk_analysis["chunks_by_source"][source_file]["valid"] += 1
        
        # Save individual chunk file
        chunk_filename = f"{chunk_id}.json"
        chunk_filepath = os.path.join(product_debug_path, chunk_filename)
        
        with open(chunk_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "chunk_id": chunk_id,
                "metadata": {
                    "product_name": obj.get('product_name', ''),
                    "doc_type": obj.get('doc_type', ''),
                    "source_file": obj.get('source_file', ''),
                    "chunk_index": i + 1,
                    "content_length": content_length,
                    "is_empty": is_empty
                },
                "content": content,
                "questions": obj.get('questions', [])
            }, f, indent=2, ensure_ascii=False)
    
    # Save summary analysis
    summary_filepath = os.path.join(product_debug_path, f"{product_name}_chunk_analysis.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(chunk_analysis, f, indent=2, ensure_ascii=False)
    
    # Save human-readable summary
    summary_txt_filepath = os.path.join(product_debug_path, f"{product_name}_summary.txt")
    with open(summary_txt_filepath, 'w', encoding='utf-8') as f:
        f.write(f"CHUNK ANALYSIS SUMMARY FOR {product_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Chunks: {chunk_analysis['total_chunks']}\n")
        f.write(f"Empty Chunks: {len(chunk_analysis['empty_chunks'])}\n")
        f.write(f"Valid Chunks: {len(chunk_analysis['valid_chunks'])}\n\n")
        
        f.write("BREAKDOWN BY DOCUMENT TYPE:\n")
        f.write("-" * 30 + "\n")
        for doc_type, stats in chunk_analysis["chunks_by_doc_type"].items():
            f.write(f"{doc_type}: {stats['total']} total, {stats['valid']} valid, {stats['empty']} empty\n")
        
        f.write("\nBREAKDOWN BY SOURCE FILE:\n")
        f.write("-" * 30 + "\n")
        for source_file, stats in chunk_analysis["chunks_by_source"].items():
            f.write(f"{source_file}: {stats['total']} total, {stats['valid']} valid, {stats['empty']} empty\n")
        
        if chunk_analysis['empty_chunks']:
            f.write("\nEMPTY CHUNKS:\n")
            f.write("-" * 15 + "\n")
            for chunk_id in chunk_analysis['empty_chunks']:
                f.write(f"- {chunk_id}\n")
    
    logger.info(f"Debug analysis saved to: {summary_filepath}")
    logger.info(f"Found {len(chunk_analysis['empty_chunks'])} empty chunks out of {chunk_analysis['total_chunks']} total chunks")
    
    return chunk_analysis


def embed_product(product_name, weaviate_client):
    """
    Processes documents for a single product and embeds them into the 'Insurance_Knowledge_Base' collection using raw Weaviate.
    """
    logger.info(f"Starting embedding process for product: {product_name}")
    
    collection_name = "Insurance_Knowledge_Base"
    
    try:
        docs_collection = weaviate_client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"Failed to get Weaviate collection '{collection_name}': {e}")
        return

    # Define file paths and their corresponding chunking functions and doc_types
    files_to_process = [
        {"path": os.path.join(SOURCE_DB_PATH, "benefits", f"{product_name}_benefits.txt"), "chunker": chunk_benefits, "doc_type": "benefits"},
        {"path": os.path.join(SOURCE_DB_PATH, "FAQ", f"{product_name}_FAQs.txt"), "chunker": chunk_faqs, "doc_type": "faq"},
        {"path": os.path.join(SOURCE_DB_PATH, "policy", f"{product_name}_policy.md"), "chunker": chunk_policy_md, "doc_type": "policy"},
    ]

    all_objects = []
    
    for file_info in files_to_process:
        file_path = file_info["path"]
        if os.path.exists(file_path):
            logger.info(f"Processing file: {file_path}")
            chunks = file_info["chunker"](file_path)
            logger.info(f"Generated {len(chunks)} chunks from {os.path.basename(file_path)}")
            
            valid_chunks_count = 0
            empty_chunks_count = 0
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{product_name}_{file_info['doc_type']}_{i+1}"
                
                if not chunk or not chunk.strip():
                    logger.warning(f"Skipping empty chunk {chunk_id} from file: {os.path.basename(file_path)}")
                    empty_chunks_count += 1
                    continue

                valid_chunks_count += 1
                logger.debug(f"Processing valid chunk {chunk_id}: {len(chunk)} characters")

                # Generate hypothetical questions for each chunk
                questions = generate_hypothetical_questions(chunk)
                
                all_objects.append({
                    "content": chunk,
                    "questions": questions, # Add generated questions
                    "product_name": product_name,
                    "doc_type": file_info["doc_type"],
                    "source_file": os.path.basename(file_path)
                })
            
            logger.info(f"File {os.path.basename(file_path)} processing complete: {valid_chunks_count} valid chunks, {empty_chunks_count} empty chunks skipped")
        else:
            logger.warning(f"File not found, skipping: {file_path}")

    if not all_objects:
        logger.warning(f"No chunks generated for product: {product_name}")
        return

    # Save chunks to debug folder for analysis
    logger.info(f"Generated {len(all_objects)} data objects for product: {product_name}")
    chunk_analysis = save_chunks_to_debug_folder(product_name, all_objects)
    
    # Log analysis summary
    logger.info(f"Chunk Analysis Summary - Total: {chunk_analysis['total_chunks']}, Valid: {len(chunk_analysis['valid_chunks'])}, Empty: {len(chunk_analysis['empty_chunks'])}")
    
    # Only proceed with Weaviate ingestion if we have valid chunks
    valid_objects = [obj for obj in all_objects if obj.get('content') and obj.get('content').strip()]
    
    if not valid_objects:
        logger.error(f"No valid chunks found for product: {product_name}. Skipping Weaviate ingestion.")
        return
    
    if len(valid_objects) != len(all_objects):
        logger.warning(f"Filtering out {len(all_objects) - len(valid_objects)} empty chunks before Weaviate ingestion")
    
    logger.info(f"Ingesting {len(valid_objects)} valid chunks into Weaviate...")

    try:
        logger.info(f"Generating embeddings and ingesting {len(valid_objects)} objects...")
        
        with docs_collection.batch.dynamic() as batch:
            for i, item in enumerate(valid_objects):
                try:
                    # Generate SEPARATE embeddings for content and questions
                    content_text = item.get('content', '')
                    questions_text = ' '.join(item.get('questions', []))
                    
                    if content_text:
                        # Generate separate embeddings (3072-dim each)
                        logger.debug(f"Generating content embedding for chunk {i+1}")
                        content_embedding = embedding_model.embed_query(content_text)
                        
                        # Prepare vectors dictionary
                        vectors = {"content_vector": content_embedding}
                        
                        # Generate questions embedding if questions exist
                        if questions_text and questions_text.strip():
                            logger.debug(f"Generating questions embedding for chunk {i+1}")
                            questions_embedding = embedding_model.embed_query(questions_text)
                            vectors["questions_vector"] = questions_embedding
                        
                        # Store with named vectors
                        batch.add_object(
                            properties=item,  # Same metadata structure
                            vector=vectors    # Multiple named vectors
                        )
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1}/{len(valid_objects)} objects with embeddings")
                    else:
                        logger.warning(f"Skipping object {i+1} - no content to embed")
                        
                except Exception as embed_error:
                    logger.error(f"Failed to generate embedding for object {i+1}: {embed_error}")
                    # Add without embedding as fallback
                    batch.add_object(properties=item)
        
        if docs_collection.batch.failed_objects:
             logger.error(f"Failed to ingest {len(docs_collection.batch.failed_objects)} objects for {product_name}.")
             for failed_obj in docs_collection.batch.failed_objects:
                 logger.error(f"Failed object: {failed_obj}")
        
        logger.info(f"Successfully ingested {len(valid_objects)} valid objects for {product_name} into Weaviate collection: {collection_name}")

    except Exception as e:
        logger.error(f"Failed to ingest documents for {product_name}: {e}")
        
    # Log final summary
    logger.info(f"Embedding process completed for {product_name}:")
    logger.info(f"  - Total chunks processed: {chunk_analysis['total_chunks']}")
    logger.info(f"  - Valid chunks ingested: {len(valid_objects)}")
    logger.info(f"  - Empty chunks filtered out: {len(chunk_analysis['empty_chunks'])}")
    logger.info(f"  - Debug files saved to: {os.path.join(DEBUG_OUTPUT_PATH, product_name)}")

def main():
    """
    Main function to parse arguments and trigger the embedding process.
    """
    parser = argparse.ArgumentParser(description="Embedding agent for processing product documents.")
    parser.add_argument("--product", type=str, help="The name of the product to process. If not provided, all products will be processed.")
    args = parser.parse_args()

    # Create debug output directory
    os.makedirs(DEBUG_OUTPUT_PATH, exist_ok=True)
    logger.info(f"Debug output will be saved to: {DEBUG_OUTPUT_PATH}")

    from utils.weaviate_client import get_weaviate_client
    client = None
    try:
        client = get_weaviate_client()
        collection_name = "Insurance_Knowledge_Base"

        # Idempotent collection creation
        if not client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist. Creating it now.")
            client.collections.create(
                name=collection_name,
                # Configure NAMED VECTORS for separate semantic spaces (MANUAL EMBEDDINGS)
                vector_config=[
                    {
                        "name": "content_vector",
                        "vectorizer": Configure.Vectorizer.none(),  # Manual embeddings
                        "vector_index_config": Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        )
                    },
                    {
                        "name": "questions_vector",
                        "vectorizer": Configure.Vectorizer.none(),  # Manual embeddings
                        "vector_index_config": Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE
                        )
                    }
                ],
                properties=[
                    # KEEP EXACT SAME METADATA (for rec_retriever_agent compatibility)
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="questions", data_type=DataType.TEXT_ARRAY),
                    Property(name="product_name", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="source_file", data_type=DataType.TEXT),
                ],
            )
            logger.info(f"Successfully created collection: {collection_name}")
        else:
            logger.info(f"Using existing Weaviate collection: {collection_name}")

        if args.product:
            embed_product(args.product, client)
        else:
            logger.info("No specific product specified. Processing all available products.")
            products = get_all_products()
            if not products:
                logger.warning("No product directories found in source_db.")
                return
            for product in products:
                embed_product(product, client)
    
    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}")
    finally:
        if client and client.is_connected():
            client.close()
            logger.info("Weaviate client closed.")

if __name__ == "__main__":
    main()
