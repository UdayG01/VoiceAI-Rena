import json
import os
import smtplib
import uuid
import time # Added for the interactive loop's exception handling
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
from langchain_core.messages import HumanMessage # Added for agent input

# Set device and suppress Sentence Transformer warnings on import
device = "cpu"

load_dotenv()
logger.remove() # Remove default loguru configuration
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# --- Path Configuration for Portability ---
# CRITICAL FIX: Use relative paths based on the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_INTEGRATION_DIR = os.path.join(BASE_DIR, "rag_integration")
DATA_FILE_PATH = os.path.join(BASE_DIR, "renata_data.json")

INDEX_FILE_PATH = os.path.join(RAG_INTEGRATION_DIR, r"D:\RenataVoiceAI\VoiceAI-Rena\src\rag_integration\company_rag_index.bin")
CORPUS_FILE_PATH = os.path.join(RAG_INTEGRATION_DIR, r"D:\RenataVoiceAI\VoiceAI-Rena\src\rag_integration\company_corpus_chunks.npy")

RAG_INDEX = None
RAG_CORPUS = None
RAG_EMBEDDER = None
try:
    # 'all-MiniLM-L6-v2' is the model used in create_index.py
    logger.info("⏳ Loading RAG components (Sentence Transformer, FAISS index, Corpus)...")
    RAG_EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    RAG_INDEX = faiss.read_index(INDEX_FILE_PATH)
    RAG_CORPUS = np.load(CORPUS_FILE_PATH, allow_pickle=True) # Added allow_pickle=True for safety
    logger.info(f"✅ RAG components loaded. Index dimension: {RAG_INDEX.d}, Chunks: {len(RAG_CORPUS)}")
except FileNotFoundError as e:
    logger.error(f"❌ RAG files not found at {e.filename}. Please ensure they are in the 'rag_integration' subdirectory.")
    RAG_INDEX, RAG_CORPUS, RAG_EMBEDDER = None, None, None
except Exception as e:
    logger.error(f"❌ Error loading RAG components: {e}. The RAG search tool will not function.")


"""
# TOOLS FOR THE AGENT
- RAG Search Tool: Retrieves relevant company info from the knowledge base.
- Complaint Registration Tool: Gathers complaint details and sends confirmation email.
"""

def load_company_data():
    """Loads company data using the new relative path."""
    if not os.path.exists(DATA_FILE_PATH):
        logger.warning(f"renata_data.json not found at {DATA_FILE_PATH}. Company data for complaint logic might be incomplete.")
        return {} # Return empty dict instead of raising FileNotFoundError to let agent run

    try:
        with open(DATA_FILE_PATH, "r") as f:
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
        # Filter out invalid indices (-1 can appear if k is larger than the number of vectors)
        valid_indices = [i for i in I[0] if i != -1 and i < len(RAG_CORPUS)]
        retrieved_chunks = [RAG_CORPUS[i] for i in valid_indices]
        
        if not retrieved_chunks:
             return "No relevant documents found in the knowledge base for that query."

        # 4. Concatenate and return the context string for the LLM
        return "---".join(retrieved_chunks)

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return "ERROR: An error occurred during the knowledge base search."


# --- 3. Complaint Handling Tools (Updated with Sanitization) ---
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
    # --- CRITICAL FIX: Email Sanitization for STT errors ---
    sanitized_email = email.lower()
    
    # 1. Replace spoken terms for the dot (.)
    sanitized_email = sanitized_email.replace(" dot ", ".").replace(" point ", ".").replace(" full stop ", ".")
    
    # 2. Replace spoken terms for the at (@)
    sanitized_email = sanitized_email.replace(" at ", "@").replace(" a t ", "@")
    
    # 3. Remove all spaces
    sanitized_email = "".join(sanitized_email.split())
    
    # Use the sanitized email unless it looks completely broken
    final_email = sanitized_email if "@" in sanitized_email and "." in sanitized_email else email
    # --------------------------------------------------------

    ticket_id = "TKT-" + uuid.uuid4().hex[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    complaint_record = {
        "ticket_id": ticket_id,
        "name": name,
        "email": final_email, # Use the sanitized email
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

ID Ticket: {ticket_id}
Submitted: {timestamp}

Issue Summary:
{issue}

One of our support specialists will contact you shortly.
If you need to add more details, simply reply to this email with your ticket ID.

Best regards,
Renata Support Team
"""

    # Send confirmation email
    send_email(final_email, f"Complaint Registered - {ticket_id}", email_body)

    return {
        "message": "Complaint submitted and confirmation email sent.",
        "ticket_id": ticket_id,
        "recorded_issue": issue,
        "contact": final_email
    }


"""Tool List and System Prompt Update (Unchanged)"""

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
**Tool Output**: {"result": "Renata was founded by Anil Sagar in 2018."}
**Assistant**: Renata was founded by Anil Sagar in 2018.

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
# You can uncomment and use Groq or OpenAI if you wish to test non-local LLMs
# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     max_tokens=256,
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


# --- Interactive Console Loop (NEW BLOCK) ---

if __name__ == "__main__":
    logger.info("Agent initialized. Starting interactive console loop.")
    print("-------------------------------------------------------")
    print(f"Renata AI Support Agent (Using Ollama model: {model.model})")
    print("-------------------------------------------------------")
    print("Type 'exit' or 'quit' to stop.")
    
    # CRITICAL: Check Ollama connectivity before starting loop
    try:
        # A quick ping to the Ollama endpoint
        model.invoke("Hello, check connectivity.") 
        print("✅ Ollama is reachable and model is loaded.")
    except Exception as e:
        print("\n-------------------------------------------------------")
        print("❌ CRITICAL ERROR: OLLAMA MODEL UNAVAILABLE.")
        print("-------------------------------------------------------")
        print(f"Details: {e}")
        print("Please ensure the Ollama server is running and the model (qwen3:1.7b) is pulled.")
        print("Run: `ollama run qwen3:1.7b` in a separate terminal before running this script.")
        exit()

    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting agent. Goodbye!")
                break
            
            if not user_input.strip():
                continue
            
            # Prepare input for the agent (LangGraph/LangChain typically expects a list of messages)
            messages = [HumanMessage(content=user_input)]

            # Invoke the agent (start the graph execution)
            # We use the agent object directly as the runnable
            response = agent.invoke(
                input={"messages": messages}, # Input key is "messages" for the ReAct Agent Graph
                config=agent_config
            )

            # The final response is the last message in the list
            # We take the content of the last message part.
            agent_output = response["messages"][-1].content
            
            # Print the final response
            print(f"Renata: {agent_output}")

        except KeyboardInterrupt:
            print("\nExiting agent. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred during console interaction: {e}")
            print("Renata: I apologize, an internal processing error occurred. Please try again.")
