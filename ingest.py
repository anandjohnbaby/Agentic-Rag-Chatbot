import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
VECTOR_DB_DIR = "vectordb"

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def ingest_topic(topic: str):
    # Loads scraped text, chunks it, embeds it, and stores in ChromaDB.
    topic_clean = topic.replace(" ", "_")
    file_path = os.path.join(DATA_DIR, f"{topic_clean}.txt")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Scraped data not found. Run scraper first.")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    metadatas = [{"topic": topic, "source": "Wikipedia"} for _ in chunks]

    Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=VECTOR_DB_DIR
    )
    