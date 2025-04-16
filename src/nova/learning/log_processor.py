# --- Tell ChromaDB to use pysqlite3 binary ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -------------------------------------------

import os
import logging
import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
LOG_FILE_PATH = "logs/interaction_log.jsonl"
DB_PATH = "db/chroma_db"
COLLECTION_NAME = "interaction_history"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Ensure DB directory exists
os.makedirs(DB_PATH, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_logs():
    logger.info("Starting log processing...")

    # --- Initialize ChromaDB ---
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        # Use get_or_create_collection to avoid errors if collection already exists
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine distance for embeddings
        )
        logger.info(f"Connected to ChromaDB. Collection '{COLLECTION_NAME}' ready.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
        return

    # --- Initialize Embedding Model ---
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        return

    # --- Process Log File ---
    processed_count = 0
    added_count = 0
    if not os.path.exists(LOG_FILE_PATH):
        logger.error(f"Log file not found: {LOG_FILE_PATH}")
        return

    logger.info(f"Reading interaction log file: {LOG_FILE_PATH}")
    try:
        with open(LOG_FILE_PATH, 'r') as f:
            for line in f:
                processed_count += 1
                try:
                    log_entry = json.loads(line.strip())

                    # Ensure necessary fields are present and outcome is SUCCESS
                    # Check outcome specifically AFTER checking key existence
                    if all(k in log_entry for k in ['session_id', 'step_id', 'simplified_state', 'user_goal', 'action_taken', 'outcome']): 
                        if log_entry['outcome'] == 'SUCCESS':
                            state = log_entry['simplified_state']
                            goal = log_entry['user_goal']
                            action = log_entry['action_taken'] # Assuming action is already a string/JSON string

                            # Generate unique ID for ChromaDB entry
                            entry_id = f"{log_entry['session_id']}_{log_entry['step_id']}"

                            # --- Generate Embedding ---
                            # Embed the state, potentially combining with goal for relevance?
                            # For now, just embedding the state description.
                            if not state: # Handle empty state string
                                logger.warning(f"Skipping entry {entry_id} due to empty state.")
                                continue
                                
                            try:
                                embedding = model.encode(state).tolist()
                            except Exception as emb_e:
                                logger.warning(f"Failed to generate embedding for state: {state[:100]}... Error: {emb_e}", exc_info=True)
                                continue # Skip this entry if embedding fails

                            # --- Add to ChromaDB ---
                            try:
                                collection.add(
                                    embeddings=[embedding],
                                    metadatas=[{
                                        "goal": goal,
                                        "action": action, # Store the successful action
                                        "outcome": "SUCCESS",
                                        "state_text": state # Store original state text for reference
                                    }],
                                    ids=[entry_id]
                                )
                                added_count += 1
                                logger.debug(f"Added entry to ChromaDB: {entry_id}")
                            except chromadb.errors.IDAlreadyExistsError:
                                logger.debug(f"Entry ID {entry_id} already exists in ChromaDB. Skipping.")
                            except Exception as db_e:
                                logger.error(f"Failed to add entry {entry_id} to ChromaDB: {db_e}", exc_info=True)

                        elif log_entry.get('outcome') != 'SUCCESS':
                            logger.debug(f"Skipping non-SUCCESS entry: {log_entry.get('session_id', 'N/A')}_{log_entry.get('step_id', 'N/A')}")
                    else:
                        logger.warning(f"Skipping incomplete log entry (missing keys): {log_entry}")

                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line.strip()}")
                except Exception as entry_e:
                    logger.error(f"Error processing log entry: {entry_e}", exc_info=True)

    except FileNotFoundError:
        logger.error(f"Log file not found at path: {LOG_FILE_PATH}")
    except Exception as file_e:
        logger.error(f"Error reading log file: {file_e}", exc_info=True)

    logger.info(f"Log processing finished. Processed {processed_count} lines.")
    logger.info(f"Added {added_count} successful interactions to ChromaDB collection '{COLLECTION_NAME}'.")
    # Log current count in collection for verification
    try:
        count = collection.count()
        logger.info(f"Total entries currently in collection '{COLLECTION_NAME}': {count}")
    except Exception as e:
        logger.warning(f"Could not retrieve count from collection '{COLLECTION_NAME}': {e}")


if __name__ == "__main__":
    process_logs() 