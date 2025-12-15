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
logger.remove() # Remove default loguru configuration
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)


# --- 1. RAG Index Configuration & Loading ---

INDEX_FILE_PATH = "src/rag_integration/company_rag_index_2.bin"
CORPUS_FILE_PATH = "src/rag_integration/company_corpus_chunks_2.npy"
METADATA_FILE_PATH = "src/rag_integration/company_corpus_metadata_2.npy"

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


# --- 2. RAG Search Tool (Hybrid + Diversified) ---
@tool
def rag_search(query: str, k: int = 3) -> str:
    """
    Hybrid RAG search using vector similarity + BM25 keyword matching,
    with increased candidate retrieval (Fix 1) and section diversification (Fix 2).
    """
    if (
        RAG_INDEX is None
        or RAG_CORPUS is None
        or RAG_EMBEDDER is None
        or BM25_INDEX is None
        or RAG_METADATA is None
    ):
        return "ERROR: RAG system is currently unavailable."

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
        initial_k = min(6, len(RAG_CORPUS))  # candidate pool
        final_k = k                            # tool API remains unchanged

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
            hybrid_scores[idx] = (0.7 * vs) + (0.3 * bs)

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
        return "---".join(retrieved_chunks)

    except Exception as e:
        logger.error(f"Hybrid RAG search failed: {e}")
        return "ERROR: An error occurred during the knowledge base search."



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

    return {
        "message": "Complaint submitted and confirmation email sent.",
        "ticket_id": ticket_id,
        "recorded_issue": issue,
        "contact": email
    }


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
#   model="x-ai/grok-4.1-fast:free",
#   max_tokens=256
# )

model = ChatOllama(
    model="qwen3:1.7b",
    temperature=0,
    max_tokens=128,
)
memory = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools, # This list now contains [rag_search, register_complaint]
    prompt=system_prompt,
    checkpointer=memory,
)

agent_config = {"configurable": {"thread_id": "default_user"}}

