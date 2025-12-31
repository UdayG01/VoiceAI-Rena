import json
import os
import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# RAG & LLM dependencies
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_agent
from langchain.tools import tool
from rank_bm25 import BM25Okapi

from loguru import logger
from dotenv import load_dotenv

# Set device and suppress Sentence Transformer warnings on import
device = "cpu"
# You might need to add: import torch; torch.set_default_device(device) 
# if you run into tensor/device issues, but usually not needed for loading.

load_dotenv()

# Configure logging to markdown files
import os
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Generate timestamp for log filename
from datetime import datetime
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"rag_session_{log_timestamp}.md"

logger.remove()  # Remove default loguru configuration

# Add file logger with markdown-friendly format (ALL details go here)
logger.add(
    LOG_FILE,
    format="**{time:HH:mm:ss}** | `{level}` | {message}",
    level="DEBUG",
    mode="w",
    encoding="utf-8"
)

# Add console logger (basic INFO like before, but exclude detailed tool logs)
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    level="INFO",
    filter=lambda record: "🔧" not in record["message"] and "🧠 Agent Input:" not in record["message"] and "🧠 Agent returned" not in record["message"] and "🧠 Raw Agent Response:" not in record["message"]
)

logger.info(f"📝 Session logs saved to: logs/{LOG_FILE.name}")

# Write markdown header to log file
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(f"""# RAG Voice Agent Session Log
**Session Started:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Log File:** `{LOG_FILE.name}`

---

## Session Activity

""")

logger.debug(f"Initialized RAG session log at {LOG_FILE}")


# --- 1. RAG Index Configuration & Loading ---

INDEX_FILE_PATH = "src/rag_integration/company_rag_index_3.bin"
CORPUS_FILE_PATH = "src/rag_integration/company_corpus_chunks_3.npy"
METADATA_FILE_PATH = "src/rag_integration/company_corpus_metadata_3.npy"

RAG_INDEX = None
RAG_CORPUS = None
RAG_METADATA = None
RAG_EMBEDDER = None
BM25_INDEX = None
BM25_TOKENIZED_CORPUS = None

try:
    logger.info("⏳ Loading RAG components (Sentence Transformer, FAISS index, Corpus)...")

    RAG_EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    RAG_INDEX = faiss.read_index(INDEX_FILE_PATH)
    RAG_CORPUS = np.load(CORPUS_FILE_PATH, allow_pickle=True)
    RAG_METADATA = np.load(METADATA_FILE_PATH, allow_pickle=True)


    # --- NEW: BM25 setup ---
    BM25_TOKENIZED_CORPUS = [
        chunk.lower().split() for chunk in RAG_CORPUS
    ]
    BM25_INDEX = BM25Okapi(BM25_TOKENIZED_CORPUS)

    logger.info(
        f"✅ RAG loaded | Chunks: {len(RAG_CORPUS)} | "
        f"Index dim: {RAG_INDEX.d}"
    )

except Exception as e:
    logger.error(f"❌ Error loading RAG components: {e}")



"""
# TOOLS FOR THE AGENT
- RAG Search Tool: Retrieves relevant company info from the knowledge base.
- Complaint Registration Tool: Gathers complaint details and sends confirmation email.
"""
# The load_company_data function is updated to look for 'renata_data.json'.
def load_company_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # --- CHANGED to renata_data.json ---
    file_path = os.path.join(base_dir, "renata_data.json") 

    if not os.path.exists(file_path):
        # --- CHANGED to renata_data.json ---
        logger.warning(f"renata_data.json not found at {file_path}. Company data for complaint logic might be incomplete.")
        return {} # Return empty dict instead of raising FileNotFoundError to let agent run

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("renata_data.json is not valid JSON.")
        return {}
    
COMPANY_DATA = load_company_data()

# --- Global Context for Query Validation ---
# This stores the original user query so rag_search can validate LLM reformulations
_original_user_query = None


def set_user_query(query: str):
    """Store the original user query for validation."""
    global _original_user_query
    _original_user_query = query


# --- Query Validation Helper (Option 2 Safeguard) ---
def extract_keywords(query: str) -> list:
    """
    Extract important keywords from user query.
    These are terms we don't want the LLM to lose during reformulation.
    """
    # Common stop words to ignore
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'must', 'about', 'tell', 'me', 'what', 'how', 'why',
        'when', 'where', 'which', 'who', 'its', 'it', 'this', 'that', 'these', 'those'
    }
    
    # Extract words, convert to lowercase, filter stop words
    words = query.lower().split()
    keywords = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in stop_words]
    
    # Keep meaningful terms (length > 2)
    keywords = [k for k in keywords if len(k) > 2]
    
    return keywords


def validate_rag_query(original_query: str, llm_query: str) -> str:
    """
    Validate that LLM didn't corrupt the query intent.
    If key terms are lost, fall back to original query.
    """
    # Extract important keywords from original
    key_terms = extract_keywords(original_query)
    
    # If original query is very short, just use it directly
    if len(key_terms) <= 2:
        logger.debug(f"🔍 Short query, using original: '{original_query}'")
        return original_query
    
    # Check if LLM query preserved at least 50% of key terms
    llm_lower = llm_query.lower()
    preserved_count = sum(1 for term in key_terms if term in llm_lower)
    preservation_rate = preserved_count / len(key_terms) if key_terms else 0
    
    if preservation_rate < 0.5:
        logger.warning(
            f"🔍 Query validation FAILED: {preservation_rate:.0%} keywords preserved\n"
            f"   Original: '{original_query}' (keywords: {key_terms})\n"
            f"   LLM reformulated: '{llm_query}'\n"
            f"   → Falling back to ORIGINAL query"
        )
        return original_query
    else:
        logger.debug(
            f"🔍 Query validation PASSED: {preservation_rate:.0%} keywords preserved\n"
            f"   Original: '{original_query}'\n"
            f"   LLM reformulated: '{llm_query}' ✓"
        )
        return llm_query


# --- 2. RAG Search Tool (Hybrid + Diversified) ---
@tool
def rag_search(query: str, k: int = 5) -> str:
    """
    Hybrid RAG search using vector similarity + BM25 keyword matching,
    with increased candidate retrieval (Fix 1) and section diversification (Fix 2).
    """
    global _original_user_query
    
    # Validate query against original user input if available
    if _original_user_query:
        validated_query = validate_rag_query(_original_user_query, query)
        if validated_query != query:
            logger.info(f"🔧 TOOL CALL: rag_search(query='{query}' → CORRECTED to '{validated_query}', k={k})")
            query = validated_query
        else:
            logger.info(f"🔧 TOOL CALL: rag_search(query='{query}' ✓ validated, k={k})")
    else:
        logger.info(f"🔧 TOOL CALL: rag_search(query='{query}', k={k})")
    
    if (
        RAG_INDEX is None
        or RAG_CORPUS is None
        or RAG_EMBEDDER is None
        or BM25_INDEX is None
        or RAG_METADATA is None
    ):
        error_msg = "ERROR: RAG system is currently unavailable."
        logger.error(f"🔧 TOOL RESULT: {error_msg}")
        return error_msg

    try:
        # -----------------------
        # 1. Embed the query
        # -----------------------
        query_embedding = RAG_EMBEDDER.encode(
            query,
            convert_to_tensor=False,
            device=device
        ).astype(np.float32).reshape(1, -1)

        # -----------------------
        # Fix 1: Retrieve MORE candidates than k
        # -----------------------
        initial_k = min(10, len(RAG_CORPUS))  # Increased candidate pool
        final_k = k                             # tool API remains unchanged

        distances, vector_ids = RAG_INDEX.search(query_embedding, initial_k)

        # Convert L2 distance → similarity
        vector_scores = {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in zip(vector_ids[0], distances[0])
        }

        # -----------------------
        # 2. BM25 keyword search
        # -----------------------
        query_tokens = query.lower().split()
        bm25_scores = BM25_INDEX.get_scores(query_tokens)

        # -----------------------
        # 3. Hybrid score fusion
        # -----------------------
        hybrid_scores = {}

        for idx in range(len(RAG_CORPUS)):
            vs = vector_scores.get(idx, 0.0)
            bs = bm25_scores[idx]
            # Adjusted weights: more BM25 for structured data
            hybrid_scores[idx] = (0.6 * vs) + (0.4 * bs)

        # -----------------------
        # Fix 2: Diversify by top_section metadata
        # -----------------------
        selected_indices = []
        seen_sections = set()

        for idx, score in sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            section = RAG_METADATA[idx].get("top_section")

            if section not in seen_sections:
                selected_indices.append(idx)
                seen_sections.add(section)

            if len(selected_indices) >= final_k:
                break

        # -----------------------
        # 5. Return context
        # -----------------------
        retrieved_chunks = [RAG_CORPUS[idx] for idx in selected_indices]
        result = "---".join(retrieved_chunks)
        
        # Log detailed retrieval info
        retrieved_sections = [RAG_METADATA[idx].get("top_section", "unknown") for idx in selected_indices]
        logger.info(f"🔧 TOOL RESULT: Retrieved {len(retrieved_chunks)} chunks from sections: {retrieved_sections}")
        logger.debug(f"🔧 Retrieved chunks: {retrieved_chunks}")
        
        return result

    except Exception as e:
        error_msg = f"ERROR: An error occurred during the knowledge base search: {str(e)}"
        logger.error(f"🔧 TOOL ERROR: Hybrid RAG search failed: {e}")
        logger.exception(e)
        return error_msg



# --- 3. Complaint Handling Tools (Unchanged) ---
complaint_database = {} # in-memory (can be replaced with SQLite later)

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SUPPORT_EMAIL = os.getenv("COMPANY_SUPPORT_EMAIL")


def send_email(to_email: str, subject: str, body: str):
    """Helper function to send email via SMTP. Requires environment variables."""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, SUPPORT_EMAIL]):
        logger.error("SMTP environment variables are not fully configured. Email not sent.")
        return 

    msg = MIMEMultipart()
    msg["From"] = SUPPORT_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SUPPORT_EMAIL, to_email, msg.as_string())
        server.quit()
        logger.info(f"📧 Complaint confirmation email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


@tool
def register_complaint(name: str, email: str, issue: str, phone: str = "9876543210", company: str = "RenataIOT") -> dict:
    """
    Register a customer complaint and send them a confirmation email.

    Arguments:
        name: The user's full name
        email: Customer email where confirmation will be sent
        issue: Description of the problem
        phone (optional): Customer phone number
        company (optional): Company name (if applicable)

    Returns:
        A structured dict with ticket details.
    """
    logger.info(f"🔧 TOOL CALL: register_complaint(name='{name}', email='{email}', issue='{issue}', phone='{phone}', company='{company}')")

    ticket_id = "TKT-" + uuid.uuid4().hex[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    complaint_record = {
        "ticket_id": ticket_id,
        "name": name,
        "email": email,
        "issue": issue,
        "phone": phone,
        "company": company,
        "timestamp": timestamp,
        "status": "Open"
    }

    # Save to in-memory DB
    complaint_database[ticket_id] = complaint_record

    # Email body
    email_body = f"""
Hello {name},

Thank thank you for reaching out to Renata Support.

Your complaint has been successfully registered.

🆔 Ticket ID: {ticket_id}
📅 Submitted: {timestamp}

Issue Summary:
{issue}

One of our support specialists will contact you shortly. 
If you need to add more details, simply reply to this email with your ticket ID.

Best regards,
Renata Support Team
"""

    # Send confirmation email
    send_email(email, f"Complaint Registered - {ticket_id}", email_body)

    result = {
        "message": "Complaint submitted and confirmation email sent.",
        "ticket_id": ticket_id,
        "recorded_issue": issue,
        "contact": email
    }
    
    logger.info(f"🔧 TOOL RESULT: Complaint registered - Ticket ID: {ticket_id}")
    logger.debug(f"🔧 Full result: {result}")
    
    return result


"""Tool List and System Prompt Update"""

system_prompt = """
You are Rena, the AI Support Assistant for Renata.

### SYSTEM INSTRUCTIONS - CRITICAL
1. **AUDIO-OPTIMIZED OUTPUT**: Your output is converted directly to speech.
   - **STRICTLY NO MARKDOWN**: Do NOT use bold (**text**), italics (*text*), headers (#), or code blocks.
   - **NO LISTS**: Do NOT use numbered lists (1. 2. 3.) or bullet points (-). Speak in full sentences.
   - **NO EMOTICONS**: Do NOT use emojis (e.g., 😊, 🚀).
   - **PLAIN TEXT ONLY**: Write in simple, conversational paragraphs. Instead of a list, say "First, we do X. Second, we do Y."

2. **TOOL USAGE & JSON STRUCTURE**:
   You have access to the following tools. You must use them when appropriate.
   
   **Tool: rag_search**
   - **Purpose**: Retrieve factual information about Renata.
   - **When to use**: For ANY question about the company.
   - **JSON Arguments**:
     {
       "query": "The user's question or topic",
       "k": 3
     }

   **Tool: register_complaint**
   - **Purpose**: Log a customer support ticket.
   - **When to use**: When a user reports an issue. You MUST first collect their Name, Email, and Issue description.
   - **JSON Arguments**:
     {
       "name": "User's Name",
       "email": "User's Email",
       "issue": "Description of the problem",
       "phone": "User's Phone (optional)",
       "company": "User's Company (optional)"
     }

3. **BEHAVIOR**:
   - **Concise**: Keep answers short and to the point.
   - **Factual**: Use `rag_search` for facts. Do not hallucinate.
   - **Helpful**: Guide the user through the complaint process if needed.

3a. **STRICT GROUNDING RULES** - CRITICAL:
   - ONLY state information that is EXPLICITLY in the tool results
   - DO NOT elaborate, explain, or add technical details beyond what's provided
   - If the retrieved data is a simple list, provide ONLY that list - do not explain how each item works
   - DO NOT infer processes, mechanisms, or implementation details
   - If you don't have specific information, say so - don't fill gaps with plausible-sounding content
   - When listing items, use EXACT wording from the source whenever possible
   - Example: If source says "Robotics integration with CNC machines", say exactly that - do NOT add "by fitting robotic arms and syncing tool-paths..."

4. **IMPORTANT DISTINCTIONS**:
   - RenataAI (the company) is located in Gurgaon, India
   - RenataAI serves international customers in Japan, France, Italy, Germany
   - When discussing "presence," clarify if referring to company headquarters or customer locations
   - Always cite specific client examples when available (e.g., FCC Clutch from Japan)

5. **LANGUAGE HANDLING**:
   - If the user query is in English, your response MUST be in English.
   - If the user query is in Hindi, your response MUST be in **Hinglish** (Conversational Hindi with English words).
     - Use Latin script (Romanized Hindi).
     - Keep technical terms, company names, and numbers in English.
     - Use Hindi for sentence structure, verbs, and common connectors.
     - Example: "Aapka ticket create kar diya gaya hai. Support team jald hi aapse contact karegi."
   - Do NOT translate Hindi queries into English responses.

### FEW-SHOT EXAMPLES

**User**: Who founded Renata?
**Tool Call**: (Calls `rag_search` with arguments: {"query": "Who founded Renata?"})
**Tool Output**: {"result": "Renata was founded by Anil Sagar in 2019."}
**Assistant**: Renata was founded by Anil Sagar in 2019.

**User**: I'm facing an issue with the login.
**Assistant**: I can help you with that. Could you please provide your name and email address so I can create a ticket?

**User**: My name is Alex and email is alex@test.com. The login page gives a 404 error.
**Tool Call**: (Calls `register_complaint` with arguments: {"name": "Alex", "email": "alex@test.com", "issue": "Login page gives 404 error"})
**Tool Output**: {"ticket_id": "TKT-999", "message": "Complaint registered."}
**Assistant**: I have successfully registered your complaint. Your ticket ID is TKT-999. A support specialist will reach out to you shortly. Thank you for contacting Renata Support.

**User**: What are the mechanical automation use cases?
**Tool Call**: (Calls `rag_search` with arguments: {"query": "mechanical automation use cases"})
**Tool Output**: {"result": "SOLUTIONS - mechanical automation use cases: Robotics integration with CNC machines, CNC tool offset automation, Actuator deployment, Scanner deployment"}
**Assistant**: Renata's mechanical automation use cases include robotics integration with CNC machines, CNC tool offset automation, actuator deployment, and scanner deployment.

**User**: Renata किन उद्योगों को सेवाएं प्रदान करती है?
**Tool Call**: (Calls `rag_search` with arguments: {"query": "Renata किन उद्योगों को सेवाएं प्रदान करती है?"})
**Tool Output**: {"result": "Renata serves the Automobile industry (60-70%) and other sectors like Textiles, Chemicals, and Compressors."}
**Assistant**: Renata mainly Automobile industry (60-70%) ko serve karti hai. Iske alawa, yeh Textiles, Chemicals aur Compressors jaise sectors mein bhi services provide karti hai.

**User**: Hello.
**Assistant**: Hello! I am Rena, Renata's support assistant. How can I help you today?
"""

tools = [rag_search, register_complaint]



"""
    LANGRAPH AGENT SETUP
"""
# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     max_tokens=256,
# )

# model = ChatOpenAI(
#   api_key=os.getenv("OPENROUTER_API_KEY"),
#   base_url="https://openrouter.ai/api/v1",
#   model="openai/gpt-oss-20b:free",
#   max_tokens=180  
# )

model = ChatOllama(
    model="qwen3:1.7b",
    #model="gpt-oss:20b",
    temperature=0,
    max_tokens=180,
)
memory = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools, # This list now contains [rag_search, register_complaint]
    prompt=system_prompt,
    checkpointer=memory,
)

agent_config = {"configurable": {"thread_id": "default_user"}}

