import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Vaijeenath AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0f172a;
    color: white;
}

/* Center container */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

/* Header */
.header {
    text-align:center;
    padding:20px;
    background:linear-gradient(90deg,#6366f1,#9333ea);
    border-radius:12px;
    margin-bottom:25px;
}

.header h1 {
    color:white;
    margin-bottom:5px;
}

.header p {
    color:#e0e7ff;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius:10px;
    padding:10px;
}

/* Example cards */
.example-card {
    background:#1e293b;
    padding:12px;
    border-radius:8px;
    margin-bottom:8px;
}

/* Input box */
.stChatInputContainer {
    background-color:#1e293b;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="header">
<h1>🤖 Ask Vaijeenath AI</h1>
<p>Your AI assistant for exploring my projects, skills, and experience</p>
</div>
""", unsafe_allow_html=True)

# ---------------- EXAMPLE QUESTIONS ----------------

st.markdown("### 💡 Example Questions")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="example-card">What projects has Vaijeenath Golhar built?</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-card">Explain the StudyForge AI project</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="example-card">What technologies does he use?</div>', unsafe_allow_html=True)
    st.markdown('<div class="example-card">What is his experience with RAG systems?</div>', unsafe_allow_html=True)

# ---------------- LOAD LLM ----------------

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"],
        streaming=True
    )

llm = load_llm()

# ---------------- LOAD DOCUMENTS ----------------

@st.cache_data
def load_documents():

    docs = []

    loader1 = TextLoader("portfolio_knowledge_base.txt")
    docs.extend(loader1.load())

    loader2 = TextLoader("portfolio_faq_dataset.txt")
    docs.extend(loader2.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)

documents = load_documents()

# ---------------- VECTOR DATABASE ----------------

@st.cache_resource
def load_vector_db():

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("faiss_index"):

        db = FAISS.load_local(
            "faiss_index",
            embedding,
            allow_dangerous_deserialization=True
        )

    else:

        db = FAISS.from_documents(documents, embedding)
        db.save_local("faiss_index")

    return db

vector_db = load_vector_db()

retriever = vector_db.as_retriever(
    search_kwargs={"k": 5}
)

# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_template(
"""
You are the AI assistant of Vaijeenath Golhar.

Your job is to answer questions about:

• Vaijeenath Golhar
• his projects
• his skills
• his AI work
• his education
• his experience

IMPORTANT RULES:

1. Answer ONLY using the provided context.
2. Do NOT invent information.
3. If the answer is not clearly present in the context say:
"I don't have information about that."

Context:
{context}

User Question:
{question}

Answer clearly and professionally.
"""
)

# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi 👋 I'm Vaijeenath Golhar's AI assistant. Ask me anything about his projects, skills, or experience."
        }
    ]

# ---------------- DISPLAY CHAT ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------

question = st.chat_input("Ask something about Vaijeenath...")

if question:

    st.chat_message("user").markdown(question)

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    # Retrieve documents
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    # Streaming response
    def stream():
        for chunk in llm.stream(final_prompt):
            if chunk.content:
                yield chunk.content

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # ---------------- SOURCES ----------------

    with st.expander("📚 Sources used"):
        for doc in docs:
            st.write(doc.page_content[:300])