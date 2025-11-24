import os
import logging
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Redis
import redis

# --- Configuration ---
# File paths
DATA_FILE = "src/renata_data.json"
INDEX_NAME = "renata_index"
REDIS_URL = "redis://localhost:6379"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_redis_index():
    # 1. Check if data file exists
    if not os.path.exists(DATA_FILE):
        logger.error(f"❌ Data file '{DATA_FILE}' not found.")
        return

    logger.info(f"📄 Loading data from {DATA_FILE}...")

    # 2. Define a JQ Schema for 'Meaningful Chunks'
    # This is the magic part. Instead of loading raw JSON brackets, we use JQ 
    # to transform the data into specific natural language sentences.
    # We extract:
    #  - General Company Info
    #  - Mission
    #  - Founder Details
    #  - Each Solution Vertical (as its own separate document)
    
    jq_schema = """
    . | 
    [
      "Company: " + .company_name + ". Location: " + .location + ". Industry: " + .industry + ". Founded: " + (.founded_year|tostring),
      
      "Mission: " + .mission + ". Focus Areas: " + (.focus_areas | join(", ")),
      
      "Co-Founder: " + .co_founder.name + ". Experience: " + .co_founder.experience + ". Education: " + (.co_founder.education | join(", ")),
      
      "Key Customers: " + (.customers | join(", ")),

      (.solution_verticals[] | "Solution Vertical: " + .name + ". Purpose: " + .purpose + ". Capabilities: " + (.capabilities | join(", ")))
    ] 
    | .[]
    """

    # 3. Initialize JSONLoader
    # text_content=True means we expect strings back from the JQ schema
    loader = JSONLoader(
        file_path=DATA_FILE,
        jq_schema=jq_schema,
        text_content=True
    )

    try:
        documents = loader.load()
        logger.info(f"✅ Loaded {len(documents)} meaningful documents from JSON.")
        
        # Debug: Print first chunk to verify quality
        logger.info(f"🔎 Sample Chunk: {documents[-1].page_content}")

    except Exception as e:
        logger.error(f"❌ Error loading JSON: {e}")
        logger.error("Make sure 'jq' is installed (pip install jq). On Windows, you might need Visual Studio C++ Build Tools.")
        return

    # 4. Split Text (Optional safety net)
    # Since our JQ schema already creates good paragraph-sized chunks, 
    # we use this mainly to catch any massive lists if they exist.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    logger.info(f"✂️  Split into {len(docs)} chunks for indexing.")

    # 5. Initialize Embeddings
    # Using SentenceTransformerEmbeddings as requested
    logger.info(f"🧠 Initializing SentenceTransformerEmbeddings with model: {EMBED_MODEL}...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    # 6. Create Redis Vector Store
    # This command creates the index in Redis and adds the documents
    logger.info(f"🚀 Creating Redis Vector Store '{INDEX_NAME}' at {REDIS_URL}...")
    
    try:
        # standard redis-py connection url
        Redis.from_documents(
            documents=docs,
            embedding=embeddings,
            redis_url=REDIS_URL,
            index_name=INDEX_NAME
        )
        logger.info("✅ Successfully created Redis Index!")
        logger.info("You can now query this index in your RAG agent.")
        
    except Exception as e:
        logger.error(f"❌ Failed to create Redis index: {e}")

if __name__ == "__main__":
    create_redis_index()