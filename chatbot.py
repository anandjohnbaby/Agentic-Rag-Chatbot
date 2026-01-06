import os
import streamlit as st

from scraper import scrape_wikipedia
from ingest import ingest_topic

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# ---------------- CONFIG ----------------
VECTOR_DB_DIR = "vectordb"

st.set_page_config(
    page_title="WikiRAG",
)
    
# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0b0f14;
}

/* Titles */
h1, h2, h3, h4 {
    color: #e5e7eb;
}

/* Chat bubbles */
.user-bubble {
    background-color: #1abc9c;
    color: #000;
    padding: 10px 14px;
    border-radius: 14px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 10px;
}

.bot-bubble {
    background-color: #1f2937;
    color: #e5e7eb;
    padding: 12px 14px;
    border-radius: 14px;
    max-width: 70%;
    margin-bottom: 10px;
}

/* Topic status */
.topic-ok {
    color: #22c55e;
    font-weight: 600;
}

.topic-fail {
    color: #ef4444;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- LOAD MODELS ----------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vectordb = None
if os.path.exists(VECTOR_DB_DIR):
    vectordb = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings
    )

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.3
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("##  WikiRAG")
    st.caption("Agentic Wikipedia RAG Chatbot")

    st.markdown("### Add Wikipedia Topic")
    new_topic = st.text_input(
        "e.g., Machine Learning",
        label_visibility="collapsed"
    )

    if st.button("➕ Add Topic", use_container_width=True):
        try:
            scrape_wikipedia(new_topic)
            ingest_topic(new_topic)
            st.success(f"Topic '{new_topic}' added")
            st.rerun()
        except Exception as e:
            st.error(str(e))
    
    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")

    topics = set()

    if vectordb:
        for m in vectordb.get()["metadatas"]:
            topics.add(m["topic"])

    if topics:
        for t in sorted(topics):
            st.markdown(
                f"<span class='topic-ok'>✔ {t}</span>",
                unsafe_allow_html=True
            )
    else:
        st.info("No topics added yet")


# ---------------- MAIN CHAT AREA ----------------
st.markdown("## ✨ Agentic Chat")
st.caption("Ask questions about your topics")

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(
            f"<div class='user-bubble'>{msg.content}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot-bubble'>{msg.content}</div>",
            unsafe_allow_html=True
        )

# Chat input
query = st.chat_input("Ask a question about your topics...")

if query:
    st.session_state.chat_history.append(HumanMessage(content=query))

    if not vectordb:
        answer = "No topics available. Please add a topic first."
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        docs = vectordb.similarity_search(query, k=4)

        if not docs:
            answer = (
                "I don't know. The answer is not available in the "
                "added Wikipedia topics."
            )
        else:
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
You are a Wikipedia-based assistant.

Use ONLY the context below to answer.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{query}
"""

            response = llm.invoke(prompt)
            answer = response.content

    st.session_state.chat_history.append(AIMessage(content=answer))
    st.rerun()
