import os
import json
from loguru import logger
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Redis

# ------------ CONFIG ------------
JSON_FILE = "src/renata_data.json"
REDIS_URL = "redis://localhost:6379"
INDEX_NAME = "renata_rag_index"
EMBED_MODEL = "all-MiniLM-L6-v2"
# ---------------------------------


load_dotenv()


def build_corpus_from_json(json_path: str) -> list[str]:
    """Convert structured business JSON into natural-sounding RAG-ready sentences."""

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    logger.info("🧠 Generating meaningful text chunks from structured JSON...")

    company_name = data.get("company_name")
    corpus = []

    # General overview
    corpus.append(
        f"{company_name} is a technology company based in {data['location']}, operating in {data['industry']} since {data['founded_year']}."
    )

    # Founder summary
    founder = data["co_founder"]
    corpus.append(
        f"The company was co-founded by {founder['name']}, who studied {', and '.join(founder['education'])} and has {founder['experience']}."
    )

    # Mission & focus
    corpus.append(
        f"The mission of {company_name} is {data['mission']}. Focus areas include {', '.join(data['focus_areas'])}."
    )

    # Customers
    corpus.append(
        f"The company works with clients such as {', '.join(data['customers'])}."
    )

    # Solution verticals
    for v in data["solution_verticals"]:
        corpus.append(
            f"{v['name']} focuses on {v['purpose']}."
        )
        corpus.append(
            f"This includes capabilities like: {', '.join(v['capabilities'])}."
        )

    logger.success(f"📚 Created {len(corpus)} high-quality RAG sentences.")
    return corpus


def chunk_corpus(corpus: list[str]):
    """Split longer text into semantic retrieval-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    docs = splitter.create_documents(corpus)

    logger.success(f"✂️ Chunking complete → {len(docs)} chunks.")
    return docs


def build_redis_vector_index(chunks):
    """Embed and store RAG chunks inside Redis Vector DB."""
    logger.info("🧠 Loading embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    logger.info(f"⚙️ Setting up Redis Vector Index '{INDEX_NAME}'...")

    store = Redis(
        redis_url=REDIS_URL,
        index_name=INDEX_NAME,
        embedding=embeddings,
        overwrite=True,
    )

    logger.info("🚀 Writing chunks into Redis...")
    store.add_documents(chunks)

    logger.success(f"🎯 Redis RAG index ready with {len(chunks)} embedded vectors.")

    return store


if __name__ == "__main__":
    logger.remove()
    logger.add(print, colorize=True)

    try:
        corpus = build_corpus_from_json(JSON_FILE)
        chunks = chunk_corpus(corpus)
        build_redis_vector_index(chunks)

        logger.success("\n✨ Redis Vector Index successfully created for Renata RAG.\n")
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
