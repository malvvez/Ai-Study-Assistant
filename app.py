"""
AI Study Assistant - A Retrieval-Augmented Generation (RAG) application.

This application allows users to upload PDF documents and ask questions about them.
It uses LangChain to process the documents, HuggingFace for local embeddings, 
ChromaDB as a vector store, and Google's Gemini model for generation.
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")

# Multiple PDF upload support
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    try:
        all_docs = []
        temp_paths = []

        # --- 1. Load all PDFs ---
        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                temp_paths.append(tmp_path)

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            all_docs.extend(docs)

        # --- 2. Text Splitting ---
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        splits = splitter.split_documents(all_docs)

        # --- 3. Embeddings & Vector Store ---
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # --- 4. LLM Setup ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

        # --- 5. Prompt Engineering ---
        prompt = ChatPromptTemplate.from_template("""
        You are a study assistant.

        Rules:
        - Always prioritize the provided context.
        - If the answer is fully in the context, use it.
        - If the context is incomplete, complement with your own knowledge.
        - If the context is irrelevant, answer using your own knowledge.

        - You don't need to ask for permission to use external knowledge.
        - Always try to give the best possible answer.

        - When using external knowledge, clearly indicate it.

        Context:
        {context}

        Question:
        {question}
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # --- 6. LCEL Pipeline ---
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        st.success(f"{len(uploaded_files)} PDF(s) processed successfully!")

        # --- 7. Chat Interface Management ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("What's your question?"):

            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):

                response = rag_chain.invoke(user_input)
                answer = response.content

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")

    finally:
        # --- 8. Cleanup ---
<<<<<<< HEAD
        # Always delete the temporary file to prevent memory/storage leaks, 
        # even if an error occurs during processing.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
=======
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
>>>>>>> 6edeca9 (Added support for multiple PDF uploads)
