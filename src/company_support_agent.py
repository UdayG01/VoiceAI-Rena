from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger
from langchain.tools import tool

import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.tools import tool
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

import json

def load_company_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "company_data.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ company_data.json not found at {file_path}")

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise ValueError("❌ company_data.json is not valid JSON.")

COMPANY_DATA = load_company_data()


@tool
def company_info(query: str) -> dict:
    """
    Retrieve structured company information relevant to the query.
    The LLM will format the final answer for the user.
    """

    query = query.lower()
    result = {}

    # Founder
    if "founder" in query or "co-founder" in query:
        result["founder"] = COMPANY_DATA["co_founder"]

    # General/About
    if "what do you do" in query or "about" in query or "who are you" in query:
        result["overview"] = {
            "name": COMPANY_DATA["company_name"],
            "location": COMPANY_DATA["location"],
            "industry": COMPANY_DATA["industry"],
            "mission": COMPANY_DATA["mission"],
            "customers": COMPANY_DATA["customers"]
        }

    # Mission
    if "mission" in query or "goal" in query:
        result["mission"] = COMPANY_DATA["mission"]

    # Vertical List
    if "vertical" in query or "solutions" in query or "segments" in query:
        result["verticals"] = [v["name"] for v in COMPANY_DATA["solution_verticals"]]

    # Detailed Services
    if "service" in query or "capabilities" in query or "offer" in query:
        services = []
        for v in COMPANY_DATA["solution_verticals"]:
            services.extend(v["capabilities"])
        result["services"] = services

    # Nothing matched
    if not result:
        return {"error": "NO_MATCH"}

    return result

complaint_database = {}  # in-memory (can be replaced with SQLite later)

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SUPPORT_EMAIL = os.getenv("COMPANY_SUPPORT_EMAIL")


def send_email(to_email: str, subject: str, body: str):
    """Helper function to send email via SMTP."""
    msg = MIMEMultipart()
    msg["From"] = SUPPORT_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
    server.starttls()
    server.login(SMTP_USER, SMTP_PASS)
    server.sendmail(SUPPORT_EMAIL, to_email, msg.as_string())
    server.quit()


@tool
def register_complaint(name: str, email: str, issue: str, phone: str = 9876543210, company: str = "RenataIOT") -> dict:
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

Thank you for reaching out to Renata Support.

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


tools = [company_info, register_complaint]

system_prompt = """
You are Rena (Renata AI's Support Assistant), an AI representative for Renata.

Your responsibilities:
1. Answer user questions about the company using the `company_info` tool when needed.
2. If the user is reporting a problem, gather required information politely and then call the `register_complaint` tool.
3. If the user asks something unrelated or outside your scope, politely redirect them.

Behavior Guidelines:
- Be friendly, professional, and concise.
- Use natural conversational language.
- NEVER make up company information; only use what the tool returns.
- Do NOT output the raw JSON returned by tools. Instead, convert it into a friendly formatted response.
- If tool output contains `"error"`, respond: 
  "I couldn't find information on that. Would you like me to forward this question to a support specialist?"

Tool Usage Rules:
- Only call a tool when necessary.
- Always pass tool arguments in valid JSON format with no quotes around numbers unless they are text fields.
- After a tool completes, continue speaking conversationally to the user.

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


model = ChatGroq(
    model="llama-3.1-8b-instant",
    max_tokens=512,
)

memory = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

agent_config = {"configurable": {"thread_id": "default_user"}}
