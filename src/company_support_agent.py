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
from langchain.tools import tool
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
# Using the absolute paths provided by the user
INDEX_FILE_PATH = "src/rag_integration/company_rag_index.bin"
CORPUS_FILE_PATH = "src/rag_integration/company_corpus_chunks.npy"

RAG_INDEX = None
RAG_CORPUS = None
RAG_EMBEDDER = None
try:
    # 'all-MiniLM-L6-v2' is the model used in create_index.py
    logger.info("⏳ Loading RAG components (Sentence Transformer, FAISS index, Corpus)...")
    RAG_EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device=device) 
    RAG_INDEX = faiss.read_index(INDEX_FILE_PATH)
    RAG_CORPUS = np.load(CORPUS_FILE_PATH)
    logger.info(f"✅ RAG components loaded. Index dimension: {RAG_INDEX.d}, Chunks: {len(RAG_CORPUS)}")
except Exception as e:
    logger.error(f"❌ Error loading RAG components: {e}. The RAG search tool will not function.")


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


# --- 2. RAG Search Tool (Replaces company_info) ---
@tool
def rag_search(query: str, k: int = 3) -> str:
    """
    Retrieve the most relevant context from the company's knowledge base 
    using a vector search (RAG). Use this tool for ALL factual company 
    information inquiries (founder, mission, services, customers, history, etc.).

    Args:
        query: The user's question or the core topic to search for.
        k (optional): The number of top relevant chunks to retrieve. Defaults to 3.

    Returns:
        A single string containing the retrieved context chunks, separated by '---',
        or an error message if retrieval fails.
    """
    if RAG_INDEX is None or RAG_CORPUS is None or RAG_EMBEDDER is None:
        return "ERROR: RAG system is currently unavailable or improperly loaded. Cannot search knowledge base."
        
    try:
        # 1. Embed the user question
        # Ensure encoding runs on the specified device
        query_embedding = RAG_EMBEDDER.encode(query, convert_to_tensor=False, device=device).astype(np.float32)
        query_embedding = query_embedding.reshape(1, -1) # Reshape for FAISS search

        # 2. Search the FAISS index for the top 'k' nearest neighbors
        # D is the distance array, I is the index array (IDs of the chunks)
        D, I = RAG_INDEX.search(query_embedding, k)
        
        # 3. Retrieve the actual text chunks (the context)
        retrieved_chunks = [RAG_CORPUS[i] for i in I[0]]
        
        # 4. Concatenate and return the context string for the LLM
        return "---".join(retrieved_chunks)

    except Exception as e:
        logger.error(f"RAG search failed: {e}") 
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
You are Rena (Renata AI's Support Assistant), an AI representative for Renata.

Your responsibilities:
1. Answer user questions about the company using the `rag_search` tool for factual information.
2. If the user is reporting a problem, gather required information politely and then call the `register_complaint` tool.
3. If the user asks something unrelated or outside your scope, politely redirect them.

Behavior Guidelines:
- Be friendly, professional, and concise.
- Use natural conversational language.
- NEVER make up company information; only use what the `rag_search` tool returns.
- Do NOT output the raw context returned by tools. Instead, convert it into a friendly, clear, and formatted response.
- If tool output contains **"ERROR"** or the retrieved context is insufficient, respond: 
  "I couldn't find information on that in our knowledge base. Would you like me to forward your question to a human specialist?"

Tool Usage Rules:
- Only call a tool when necessary.
- Always pass tool arguments in valid JSON format.
- After a tool completes, continue speaking conversationally to the user.

Company Information Handling Rules:
- For **ANY** question related to Renata's background, founder, mission, services, customers, industry, or details, you MUST use the `rag_search` tool.
- Pass the user's exact query (or a summarized version if complex) to the `rag_search` tool's `query` argument.
- The `rag_search` tool will return context chunks separated by '---'. Read this context carefully and use it to formulate your answer.
- Structure your answer to address all parts of the user's question, using the retrieved information as your source of truth.

Complaint Handling Flow:
- If user expresses a complaint intent (keywords: 'issue', 'problem', 'not working', 'support', 'ticket', 'complaint'):
    → Ask for missing fields: name, email, issue (and phone/company if they want to share).
- Once all required fields are collected, call the `register_complaint` tool.
- After tool returns, confirm and reassure the user politely.

Tone:
- Supportive, calm, concise, and respectful.
- Avoid technical jargon unless the user demonstrates understanding.

Your Goal:
Provide an experience similar to a real customer service representative.
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
    model="qwen3:0.6b",
    temperature=0,
    max_tokens=256,
)
memory = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools, # This list now contains [rag_search, register_complaint]
    prompt=system_prompt,
    checkpointer=memory,
)

agent_config = {"configurable": {"thread_id": "default_user"}}

# Note: The code blocks related to `fastrtc` (`response`, `create_stream`, `build_custom_ui`, `process_groq_tts`) 
# are separate utility functions that interact with this agent. They remain the same as you requested.