import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Setup environment
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def get_text_chunks(text):
    # Using RecursiveCharacterTextSplitter for better semantic splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Using a more efficient and modern embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Using HuggingFaceEndpoint for better reliability and speed (replacing deprecated Hub class)
    # Using Mistral-7B or Llama-3 for significantly better reasoning than Llama-2
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        temperature=0.5,
        max_new_tokens=512
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first!")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="OmniPDF AI", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("📚 OmniPDF AI: Multi-Doc Intelligence")
    st.markdown("---")

    # Main Chat Interface
    user_question = st.text_input("Ask a question about your documents:", placeholder="What are the key findings in the uploaded reports?")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for Document Management
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files", 
            accept_multiple_files=True, 
            type=["pdf"]
        )
        
        if st.button("🚀 Process Documents"):
            if pdf_docs:
                with st.spinner("Analyzing documents..."):
                    try:
                        # 1. Extract
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No readable text found in the uploaded PDFs.")
                            return
                        
                        # 2. Chunk
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Embed & Store
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # 4. Initialize Chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload at least one PDF file.")

        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = None
            st.session_state.conversation = None
            st.rerun()

if __name__ == '__main__':
    main()
