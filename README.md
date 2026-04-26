# 📚 Nexus PDF Chat: Multi-Document AI Intelligence

A professional RAG (Retrieval-Augmented Generation) application that allows users to upload multiple PDF documents and have an intelligent, context-aware conversation with them.

## ✨ Key Enhancements (2026 Update)

- **Modern LLM:** Upgraded from Llama-2 to **Mistral-7B-Instruct**, providing significantly better reasoning and more natural responses.
- **Smarter Splitting:** Switched to `RecursiveCharacterTextSplitter` for better semantic coherence of text chunks.
- **Robust Architecture:** Updated to `langchain_community` and `HuggingFaceEndpoint` for improved stability and speed.
- **Improved UI:** Added a refined sidebar, better error handling, and a "Clear Chat" functionality.
- **Optimized Retrieval:** Configured `k=3` retrieval to provide the most relevant context to the LLM.

## 🛠️ Technical Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **LLM:** `Mistral-7B-Instruct-v0.2` via HuggingFace Endpoint

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A HuggingFace API Token

### Installation
1. **Clone the repo**
   ```bash
   git clone https://github.com/Ravi-Tiwari-2710/Multi-pdf-ChatApp.git
   cd Multi-pdf-ChatApp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---
**Developed by [Ravi Tiwari](https://github.com/Ravi-Tiwari-2710)**
