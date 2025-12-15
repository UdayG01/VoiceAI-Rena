import json
import os
import smtplib
import uuid
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

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
from langchain_core.messages import HumanMessage

device = "cpu"
load_dotenv()
logger.remove()
logger.add(lambda msg: print(msg), colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_INTEGRATION_DIR = os.path.join(BASE_DIR, "rag_integration")
DATA_FILE_PATH = os.path.join(BASE_DIR, "renata_data.json")
INDEX_FILE_PATH = os.path.join(RAG_INTEGRATION_DIR, "company_rag_index.bin")
CORPUS_FILE_PATH = os.path.join(RAG_INTEGRATION_DIR, "company_corpus_chunks.npy")

RAG_INDEX = None
RAG_CORPUS = None
RAG_EMBEDDER = None

try:
    logger.info("⏳ Loading RAG components (Sentence Transformer, FAISS index, Corpus)...")
    RAG_EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    RAG_INDEX = faiss.read_index(INDEX_FILE_PATH)
    RAG_CORPUS = np.load(CORPUS_FILE_PATH, allow_pickle=True)
    logger.info(f"✅ RAG components loaded. Index dimension: {RAG_INDEX.d}, Chunks: {len(RAG_CORPUS)}")
except FileNotFoundError as e:
    logger.error(f"❌ RAG files not found at {e.filename}. Ensure they are in 'rag_integration'.")
    RAG_INDEX, RAG_CORPUS, RAG_EMBEDDER = None, None, None
except Exception as e:
    logger.error(f"❌ Error loading RAG: {e}")

def load_company_data():
    if not os.path.exists(DATA_FILE_PATH):
        logger.warning(f"renata_data.json not found at {DATA_FILE_PATH}.")
        return {}
    try:
        with open(DATA_FILE_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("renata_data.json is invalid JSON.")
        return {}

COMPANY_DATA = load_company_data()

@tool
def rag_search(query: str, k: int = 3) -> str:
    if RAG_INDEX is None or RAG_CORPUS is None or RAG_EMBEDDER is None:
        return "ERROR: RAG system unavailable."
    try:
        query_embedding = RAG_EMBEDDER.encode(query, convert_to_tensor=False, device=device).astype(np.float32)
        query_embedding = query_embedding.reshape(1, -1)
        D, I = RAG_INDEX.search(query_embedding, k)
        valid_indices = [i for i in I[0] if i != -1 and i < len(RAG_CORPUS)]
        retrieved_chunks = [RAG_CORPUS[i] for i in valid_indices]
        if not retrieved_chunks:
            return "No relevant documents found."
        return "---".join(retrieved_chunks)
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return "ERROR: RAG search failed."

complaint_database = {}
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SUPPORT_EMAIL = os.getenv("COMPANY_SUPPORT_EMAIL")

def send_email(to_email: str, subject: str, body: str):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, SUPPORT_EMAIL]):
        logger.error("SMTP env vars not configured.")
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
        logger.info(f"📧 Email sent to {to_email}")
    except Exception as e:
        logger.error(f"Email sending failed: {e}")

@tool
def register_complaint(name: str, email: str, issue: str,
                       phone: str = "9876543210", company: str = "RenataIOT") -> dict:
    sanitized_email = email.lower().replace(" dot ", ".").replace(" point ", ".")\
                                   .replace(" full stop ", ".").replace(" at ", "@")\
                                   .replace(" a t ", "@")
    sanitized_email = "".join(sanitized_email.split())
    final_email = sanitized_email if "@" in sanitized_email and "." in sanitized_email else email

    ticket_id = "TKT-" + uuid.uuid4().hex[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "ticket_id": ticket_id, "name": name, "email": final_email,
        "issue": issue, "phone": phone, "company": company,
        "timestamp": timestamp, "status": "Open"
    }
    complaint_database[ticket_id] = record

    email_body = f"""
Hello {name},

Thank you for reaching out to Renata Support.

Your complaint has been registered.

ID Ticket: {ticket_id}
Submitted: {timestamp}

Issue Summary:
{issue}

Our support team will contact you soon.
"""

    send_email(final_email, f"Complaint Registered - {ticket_id}", email_body)

    return {"message": "Complaint submitted.", "ticket_id": ticket_id,
            "recorded_issue": issue, "contact": final_email}

system_prompt = """
You are Rena, the AI Support Assistant for Renata.
Strict text rules...
"""

tools = [rag_search, register_complaint]

model = ChatOllama(model="qwen3:1.7b", temperature=0, max_tokens=128)
memory = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

agent_config = {"configurable": {"thread_id": "default_user"}}

if __name__ == "__main__":
    logger.info("Agent initialized.")
    print("Renata AI Support Agent")
    try:
        model.invoke("Ping")
        print("Ollama is reachable.")
    except Exception as e:
        print("ERROR: OLLAMA unavailable.", e)
        exit()

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
            messages = [HumanMessage(content=user_input)]
            response = agent.invoke({"messages": messages}, config=agent_config)
            print("Renata:", response["messages"][-1].content)
        except KeyboardInterrupt:
            print("Goodbye!")
            break
        except Exception as e:
            logger.error(f"Console error: {e}")
            print("Renata: Internal error. Please try again.")
