# 🤖 Vaijeenath AI Portfolio Assistant

An AI-powered portfolio chatbot that answers questions about my **projects, skills, experience, and AI work** using **Retrieval-Augmented Generation (RAG)**.

This application allows visitors to interact with my portfolio using natural language.

Users can ask questions like:

- What projects has Vaijeenath built?
- What skills does he have?
- Explain StudyForge AI
- What technologies does he use?

The chatbot retrieves information from my portfolio knowledge base and generates accurate responses using **Large Language Models**.

---

# 🚀 Features

- AI-powered portfolio assistant
- Retrieval-Augmented Generation (RAG)
- Semantic search using vector embeddings
- FAISS vector database
- Groq LLM integration
- Real-time streaming responses
- Built using Streamlit for an interactive UI

---

# 🧠 How It Works

1. Portfolio information is stored in text files.
2. Documents are split into chunks.
3. Embeddings are generated using HuggingFace models.
4. Embeddings are stored in a FAISS vector database.
5. When a user asks a question:
   - The system retrieves relevant documents.
   - The LLM generates an answer using the retrieved context.

Architecture Flow:

User Question  
↓  
Retriever (FAISS Vector DB)  
↓  
Relevant Context  
↓  
LLM (Groq - Llama3)  
↓  
Generated Answer  

---

# 🛠️ Tech Stack

**Programming**

- Python

**Generative AI**

- LangChain
- Retrieval-Augmented Generation (RAG)
- Groq LLM
- HuggingFace Embeddings

**Vector Database**

- FAISS

**Frontend**

- Streamlit

**Tools**

- Git
- GitHub

---

# 📂 Project Structure


vaijeenath-ai-chatbot
│
├── chatbot.py
├── portfolio_knowledge_base.txt
├── portfolio_faq_dataset.txt
├── requirements.txt
├── .gitignore
└── README.md


---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/vaijeenath-ai-chatbot.git
cd vaijeenath-ai-chatbot

Install dependencies:

pip install -r requirements.txt