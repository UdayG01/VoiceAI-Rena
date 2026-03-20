import os
import re
from datetime import datetime
from langchain_core.messages import HumanMessage
from loguru import logger

# Import the agent from your main application file
# Ensure company_support_agent.py is in the same directory or properly referenced in PYTHONPATH
try:
    from company_support_agent import agent
except ImportError:
    print("Error: Could not import 'agent' from 'company_support_agent.py'. Make sure the file exists.")
    exit(1)

# Configuration
INPUT_FILE = "src/questions.md"
OUTPUT_FILE = "src/test_results.md"

def parse_questions(filename):
    """
    Parses the markdown file to extract questions.
    Ignores headers and empty lines.
    """
    if not os.path.exists(filename):
        logger.error(f"Input file {filename} not found.")
        return []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    questions = []
    for line in lines:
        line = line.strip()
        # Skip headers, empty lines, or comments
        if not line or line.startswith("#"):
            continue
        
        # Remove numbering (e.g., "1. Who..." -> "Who...")
        clean_question = re.sub(r'^\d+\.\s*', '', line)
        questions.append(clean_question)
    
    return questions

def run_tests():
    logger.info("🚀 Starting Agent Test Harness...")
    
    questions = parse_questions(INPUT_FILE)
    if not questions:
        logger.warning("No questions found to test.")
        return

    # Prepare Output File with Header
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Renata Agent Test Results\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Questions:** {len(questions)}\n\n")
        f.write(f"---\n\n")

    print(f"\n📝 Found {len(questions)} questions. Running tests...\n")

    for idx, q in enumerate(questions, 1):
        print(f"🔹 Processing Q{idx}: {q}")
        
        # Create a unique thread ID for each question to ensure no context pollution
        # This tests the RAG retrieval capability purely.
        thread_id = f"test_run_{idx}_{int(datetime.now().timestamp())}"
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Invoke the agent
            inputs = {"messages": [HumanMessage(content=q)]}
            result = agent.invoke(inputs, config=config)
            
            # Extract the last message (AI Response)
            last_message = result["messages"][-1]
            answer = last_message.content
            
        except Exception as e:
            logger.error(f"Failed to process question '{q}': {e}")
            answer = f"❌ ERROR: Agent failed to respond. Details: {e}"

        # Append result to Markdown file
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"### Q{idx}: {q}\n")
            f.write(f"**Answer:**\n{answer}\n\n")
            f.write(f"---\n\n")

    print(f"\n✅ Testing Complete! Results saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_tests()