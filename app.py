"""
AI Study Assistant - A Retrieval-Augmented Generation (RAG) application.

This application allows users to upload a PDF document and ask questions about it.
It uses LangChain to process the document, HuggingFace for local embeddings, 
ChromaDB as a vector store, and Google's Gemini model for generation.

Author: [Seu Nome Aqui]
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain imports: Core components for our RAG pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain imports: For building the LCEL (LangChain Expression Language) chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables (e.g., Google API Key) from a .env file 
# to keep sensitive credentials secure and out of the source code.
load_dotenv()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")

# File uploader widget for the user to provide the study material
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # PyPDFLoader requires a file path, but Streamlit keeps uploaded files in memory.
    # We create a temporary file to save the PDF content to disk temporarily.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # --- 1. Document Loading ---
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # --- 2. Text Splitting ---
        # We break the document into smaller chunks to fit into the LLM's context window.
        # The 'chunk_overlap' ensures we don't lose the context if a sentence is cut in half.
        # TODO: In a more advanced version, this setup could be cached using @st.cache_resource 
        # to prevent reprocessing the PDF on every chat interaction.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        splits = splitter.split_documents(docs)

        # --- 3. Embeddings & Vector Store ---
        # Using a lightweight, open-source model from HuggingFace to generate embeddings locally.
        # This is cost-effective and faster than calling an external API for embeddings.
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Storing the embeddings in Chroma, an in-memory vector database for fast semantic search.
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        # Set up the retriever to fetch the top 5 most relevant chunks based on the user's query.
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # --- 4. LLM Setup ---
        # Initialize Google's Gemini model. Temperature is set low (0.2) to ensure 
        # the model remains factual and focuses on the provided context rather than hallucinating.
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Note: Updated to the standard 1.5-flash version
            temperature=0.2
        )

        # --- 5. Prompt Engineering ---
        # A custom prompt that instructs the LLM to act as a hybrid RAG system:
        # It prioritizes the document context but is allowed to use its own knowledge if the document lacks info.
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

        # Helper function to combine the retrieved documents into a single readable string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # --- 6. LangChain Expression Language (LCEL) Pipeline ---
        # This chain dictates the flow: Fetch context -> Format it -> Pass to Prompt -> Send to LLM
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough() # Passes the user input directly to the prompt
            }
            | prompt
            | llm
        )

        st.success("PDF processed successfully! You can now ask questions.")

        # --- 7. Chat Interface Management ---
        # Initialize the session state to keep track of the conversation history.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages in the chat interface
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle new user input
        if user_input := st.chat_input("What's your question?"):
            # Save user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate and display the assistant's response
            with st.chat_message("assistant"):
                # Invoke the RAG chain with the user's question
                response = rag_chain.invoke(user_input)
                answer = response.content

                st.markdown(answer)

                # Save assistant message to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

    except Exception as e:
        # Graceful error handling to show issues clearly in the UI
        st.error(f"An error occurred during processing: {str(e)}")

    finally:
        # --- 8. Cleanup ---
        # Always delete the temporary file to prevent memory/storage leaks, 
        # even if an error occurs during processing.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)