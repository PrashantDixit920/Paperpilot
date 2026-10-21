# 📄 PaperPilot — AI-Powered Document Research Assistant

PaperPilot is an AI-powered document research assistant built with **Python, LangChain, Gemini, Hugging Face Embeddings, ChromaDB, and Streamlit**. It enables users to upload documents, ask context-aware questions, explore relevant information with source attribution, and generate structured summaries of lengthy documents.

## ✨ Features

- **Multi-format Document Support:** Upload PDF, TXT, Markdown, and DOCX files.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document passages to ground LLM responses in the uploaded content.
- **Conversational Q&A:** Ask questions and follow-up questions using conversation history.
- **Source Attribution:** Display source filenames and page information alongside answers when available.
- **Semantic Search:** Generate local embeddings using `sentence-transformers/all-MiniLM-L6-v2` and store document chunks in ChromaDB.
- **MMR Retrieval:** Use Maximal Marginal Relevance (MMR) to retrieve relevant and diverse document chunks.
- **AI-Powered Summarization:** Summarize documents using a map-reduce pipeline designed for longer inputs.
- **Structured Outputs:** Use Pydantic schemas to organize summaries into title, overview, key findings, methodology, limitations, and conclusion.
- **Interactive Interface:** Explore document summaries, chat with documents, and inspect extracted content through Streamlit.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Google Gemini via LangChain |
| LLM Framework | LangChain |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| Retrieval Strategy | Maximal Marginal Relevance (MMR) |
| Structured Output | Pydantic |
| PDF Processing | PyPDF / LangChain document loaders |
| Package Management | uv |

## 🏗️ Architecture

PaperPilot follows a Retrieval-Augmented Generation architecture for document question answering.

```text
             User uploads documents
                      |
                      v
              Document Loader
                      |
                      v
                Text Splitting
                      |
                      v
           Hugging Face Embeddings
                      |
                      v
                  ChromaDB
                      |
                      v
User Question --> Retriever (MMR)
                      |
                      v
             Relevant Context
                      |
           Question + Chat History
                      |
                      v
                 Gemini LLM
                      |
                      v
          Answer + Source Details
```

### Document Summarization Pipeline

```text
Uploaded Documents
        |
        v
  Text Extraction
        |
        v
  Summary Chunking
        |
        v
  Map: Summarize Batches
        |
        v
  Reduce: Combine Summaries
        |
        v
  Gemini + Pydantic Schema
        |
        v
  Structured Research Summary
```

## 📁 Project Structure

```text
PaperPilot/
├── app.py
├── schema.py
├── requirements.txt
├── README.md
├── .gitignore
└── services/
    ├── document_loader.py
    ├── vector_store.py
    └── rag_chain.py
```

*The structure above represents the main application files; additional configuration files may be present.*

## ⚙️ Getting Started

### Prerequisites

- Python version compatible with the installed dependencies
- A Google Gemini API key
- Git
- `uv` for local dependency management

### 1. Clone the Repository

```bash
git clone https://github.com/PrashantDixit920/Paperpilot.git
cd paperpilot
```

### 2. Create a Virtual Environment

```bash
uv venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

If the repository includes `requirements.txt`, install the packages with:

```bash
uv pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key

Obtain an API key from [Google AI Studio](https://aistudio.google.com/).

Configure the key using an environment variable or Streamlit secrets, matching the configuration expected by the application's Gemini initialization. Never hardcode your API key or commit secrets to GitHub.

For local Streamlit secrets, create `.streamlit/secrets.toml` and use the key name expected by your code:

```toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```

If your application reads the key from an environment variable instead, configure that variable accordingly. Ensure `.streamlit/secrets.toml` is included in `.gitignore`.

### 5. Run the Application

```bash
uv run streamlit run app.py
```

Open the local URL displayed in the terminal, typically `http://localhost:8501`.


## 🔍 How to Use PaperPilot

1. Launch PaperPilot.
2. Upload one or more supported documents.
3. Process the documents to extract text, create chunks, generate embeddings, and initialize retrieval.
4. Open the **Summary** section to generate a structured summary.
5. Open the **Chat** section to ask questions about the uploaded content.
6. Ask follow-up questions to explore related concepts.
7. Review the source information associated with answers.
8. Use the **Documents** section to inspect extracted content.

## 🧠 Key Concepts Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic embeddings and vector search
- Text splitting and chunk overlap
- MMR-based document retrieval
- Prompt engineering
- Conversational context management
- LLM toolchain composition with LangChain
- Map-reduce summarization
- Pydantic structured output validation
- Streamlit application development
- Cloud deployment and API-key management

## ⚠️ Limitations

- Answer quality depends on document extraction, retrieval relevance, and LLM output quality.
- Source page information may be unavailable for some file formats.
- Conversation history helps the LLM interpret follow-up questions, but retrieval may still be less effective for questions that depend on earlier context.
- Structured summaries can occasionally fail validation if the model omits required fields.
- Uploaded documents and vector indexes are processed according to the application's current session and storage configuration; persistent storage should not be assumed.
- Gemini API usage may be subject to quotas, rate limits, and provider pricing.
- Local embedding generation consumes CPU and memory.


## 🔮 Future Improvements

- History-aware retrieval for more reliable follow-up questions.
- Persistent vector storage and document management.
- Improved citation rendering with expandable source excerpts.
- Support for additional document formats.
- Streaming responses and improved processing feedback.
- RAG evaluation and answer-quality metrics.
- Logging, tracing, and robust error recovery.
- User authentication and document-level access control.

## 👨‍💻 Author

**Prashant Dixit**

- GitHub: [PrashantDixit920](https://github.com/PrashantDixit920)
- LinkedIn: [contactprashant-dixit](https://www.linkedin.com/in/contactprashant-dixit)

