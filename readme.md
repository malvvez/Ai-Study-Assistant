# AI Study Assistant (RAG)

A simple Retrieval-Augmented Generation (RAG) application built with LangChain, Streamlit, and Google Gemini.

This project allows users to upload a PDF (e.g., lecture notes) and ask questions about it. The assistant prioritizes the document content but can complement answers using external knowledge when necessary.

---

## Objective

This project was developed to deepen my practical understanding of **Generative AI**, especially:

- Retrieval-Augmented Generation (RAG)
- Embeddings and semantic search
- Prompt engineering
- Integration with LLM APIs (Google Gemini)
- Building end-to-end AI applications

---

## How It Works

The application follows a standard RAG pipeline:

1. **PDF Upload**
   - User uploads a PDF file via Streamlit UI

2. **Document Loading**
   - The PDF is loaded using `PyPDFLoader`

3. **Text Splitting**
   - The document is split into chunks to fit LLM context limits

4. **Embeddings**
   - Each chunk is converted into vectors using a HuggingFace model (`all-MiniLM-L6-v2`)

5. **Vector Store**
   - Embeddings are stored in a Chroma in-memory database

6. **Retrieval**
   - The most relevant chunks are retrieved based on the user query

7. **LLM Generation**
   - The retrieved context is passed to a Gemini model (`gemini-1.5-flash`)
   - The model generates an answer using:
     - The document (primary source)
     - Its own knowledge (if needed)

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-study-assistant.git
cd ai-study-assistant
```

---

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Set up your API key

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY=your_api_key_here
```

You can use `.env_example` as a template.

---

### 5. Run the application

```bash
streamlit run app.py
```

---

## Optional: Check Available Models

```bash
python check_models.py
```

---

## Limitations

- Only supports **one PDF at a time**
- No persistent storage (embeddings are recalculated every run)
- No long-term memory between sessions
- Chat history is only stored temporarily (per session)
- No source citation (does not show which part of the PDF was used)
- Performance depends on local embedding computation
- Requires internet connection for LLM (Gemini API)

---

## Possible Improvements

- Support multiple PDFs
- Add source citations (show retrieved chunks)
- Cache embeddings for better performance
- Add streaming responses
- Implement conversation memory
- Improve UI/UX

---

## Tech Stack

- Python
- Streamlit
- LangChain (LCEL)
- Google Gemini API
- HuggingFace Embeddings
- ChromaDB

---

## Notes

- Embeddings are generated locally using HuggingFace (no API cost)
- LLM responses are generated using Google's Gemini API
- The system prioritizes document context but falls back to general knowledge when needed

---

## Author

[Eduardo Malves]

---

## 📄 License

This project is for educational purposes.