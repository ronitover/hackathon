# Enterprise RAG Assistant

A Streamlit chat app that lets you upload documents (PDF, DOCX, TXT), build a local semantic search index, and ask questions grounded in your files.

## Features

- Upload and manage documents from the sidebar
- FAISS vector search with BGE embeddings (runs locally)
- Grounded answers via Groq Llama 3.1 with source citations
- Incremental indexing when you add new files (full rebuild on edits/removals)

## Requirements

- Python 3.9+
- A [Groq API key](https://console.groq.com/) for answer generation
- ~500 MB disk for embedding model cache (first run downloads from Hugging Face)
- Recommended: 4 GB+ RAM

## Setup

```bash
# Clone and enter the project
cd hackathon

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=...

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Upload** one or more PDF, DOCX, or TXT files in the sidebar
2. Click **Build Knowledge Base** (first build may take a minute while the embedding model downloads)
3. **Ask questions** in the chat — answers cite passages from your documents

If you add new files later, click **Build Knowledge Base** again; only new files are embedded incrementally. Removing or replacing a file triggers a full re-index.

**Follow-up questions** are supported — the assistant remembers prior messages in the session (SQLite) and uses them when searching documents and generating answers. Refresh the page without losing chat: the session ID stays in the URL (`?session=...`).

See **[DEMO.md](DEMO.md)** for a judge-ready demo script.

## Hackathon scope

This project implements **RAG mode + conversational memory** per sprint guidance. General Chat and Web Search modes are intentionally omitted.

## Project structure

```
app.py              Streamlit UI
rag_backend.py      RAG pipeline (parse → chunk → embed → retrieve → generate)
memory.py           SQLite short-term chat memory
data/uploads/       Saved uploaded files (gitignored)
data/vectorstore/   FAISS index + manifest (gitignored)
docs/               Design mockup (reference only)
tests/              Unit tests
```

## Environment variables

| Variable       | Required | Description                    |
|----------------|----------|--------------------------------|
| `GROQ_API_KEY` | Yes      | Groq API key for LLM answers   |

## Running tests

```bash
pytest tests/ -q
```

## Troubleshooting

**First indexing is slow** — The BGE embedding model (~133 MB) downloads once from Hugging Face. Subsequent runs use the cached model.

**`GROQ_API_KEY is required`** — Copy `.env.example` to `.env` and add your key.

**No matching passages** — Try rephrasing your question, or rebuild the knowledge base after uploading new files.
