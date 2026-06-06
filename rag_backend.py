"""
RAG backend for the Enterprise RAG Assistant.

This module implements the full retrieval-augmented generation pipeline:
  upload → parse → chunk → embed → index → retrieve → generate

Dependencies (install via pip):
    langchain>=0.3.0
    langchain-community>=0.3.0
    langchain-groq>=0.2.0
    langchain-text-splitters>=0.3.0
    faiss-cpu>=1.8.0
    sentence-transformers>=3.0.0
    pypdf>=4.0.0
    docx2txt>=0.8
    python-dotenv>=1.0.0

Environment variables:
    GROQ_API_KEY  — required for answer generation (loaded from .env via python-dotenv)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load GROQ_API_KEY (and any other secrets) from .env at import time.
# Copy .env.example → .env and fill in your key before running.
load_dotenv(ENV_PATH)

# Project-relative directories for raw files and the FAISS index.
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# Embedding model: BAAI/bge-small-en-v1.5
# - Small (384-dim), fast, strong for English semantic search.
# - "bge" models are trained for retrieval; prefix queries with instruction
#   text for best results (handled in retrieve_context).
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Groq-hosted Llama 3.1 8B (replaces decommissioned llama3-8b-8192).
LLM_MODEL = "llama-3.1-8b-instant"

# Supported file extensions and their LangChain loaders.
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

# Default chunking parameters (tune per document type if needed).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Number of chunks to retrieve per query.
TOP_K = 4

# Max L2 distance (normalized embeddings) — chunks above this are treated as irrelevant.
MAX_RETRIEVAL_DISTANCE = 1.15

# BGE retrieval instruction — improves query-document matching for this model family.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# 1. Save uploaded files
# ---------------------------------------------------------------------------


def save_uploaded_files(
    uploaded_files: list[Any],
    upload_dir: Union[str, Path] = UPLOAD_DIR,
) -> list[Path]:
    """
    Persist user-uploaded files to disk so they can be parsed and indexed.

    Streamlit's file_uploader returns in-memory objects that disappear when the
    session ends. Saving to disk is the first step toward a durable knowledge
    base: the same files can be re-loaded, re-chunked, and re-indexed later.

    Args:
        uploaded_files: Iterable of file-like objects (e.g. Streamlit UploadedFile)
                        with `.name` and `.getbuffer()` or `.read()`.
        upload_dir: Directory where raw files are stored.

    Returns:
        List of absolute paths to the saved files.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for uploaded in uploaded_files:
        filename = uploaded.name
        dest = upload_dir / filename

        # Read bytes from Streamlit UploadedFile or any BinaryIO wrapper.
        if hasattr(uploaded, "getbuffer"):
            data = uploaded.getbuffer()
        else:
            uploaded.seek(0)
            data = uploaded.read()

        dest.write_bytes(data)
        saved_paths.append(dest)

    return saved_paths


# ---------------------------------------------------------------------------
# 2. Load & parse documents
# ---------------------------------------------------------------------------


def load_documents(
    upload_dir: Union[str, Path] = UPLOAD_DIR,
) -> list[Document]:
    """
    Load and parse all supported documents from the upload directory.

    DOCUMENT PARSING
    ----------------
    Parsing converts binary files (PDF, DOCX, TXT) into LangChain ``Document``
    objects — plain text plus metadata (source path, page number for PDFs).

    Each loader handles format-specific extraction:
      - PyPDFLoader: extracts text page-by-page from PDF; metadata includes ``page``.
      - Docx2txtLoader: extracts plain text from Word .docx files.
      - TextLoader: reads .txt files as UTF-8 text.

    The output is a list of Documents (often one per page for PDFs). These are
    still too large for embedding as-is — split_documents() breaks them into
    smaller chunks next.

    Args:
        upload_dir: Folder containing previously saved uploads.

    Returns:
        List of LangChain Document objects with ``page_content`` and ``metadata``.

    Raises:
        ValueError: If a file has an unsupported extension.
    """
    upload_dir = Path(upload_dir)
    if not upload_dir.exists():
        return []

    documents: list[Document] = []

    for file_path in sorted(upload_dir.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        loader_cls = LOADER_MAP.get(suffix)

        if loader_cls is None:
            raise ValueError(
                f"Unsupported file type '{suffix}' for '{file_path.name}'. "
                f"Supported: {', '.join(LOADER_MAP)}"
            )

        # Each loader reads the file and returns one or more Document objects.
        loader = loader_cls(str(file_path))
        documents.extend(loader.load())

    return documents


# ---------------------------------------------------------------------------
# 3. Chunk documents
# ---------------------------------------------------------------------------


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split parsed documents into smaller overlapping chunks.

    CHUNKING
    --------
    LLMs and embedding models have limited context windows. We cannot embed an
    entire 50-page PDF as one vector — semantic search would be too coarse and
    irrelevant sections would dominate.

    RecursiveCharacterTextSplitter:
      - Tries to split on paragraph breaks (\\n\\n), then lines, then spaces.
      - Keeps chunks around ``chunk_size`` characters with ``chunk_overlap``
        characters shared between neighbours so sentences at boundaries are not
        cut off abruptly.
      - Preserves each chunk's metadata (source file, page) for citations.

    Smaller chunks → more precise retrieval but less surrounding context.
    Larger chunks → more context per hit but noisier similarity scores.
    800 / 150 is a practical default for enterprise docs.

    Args:
        documents: Raw documents from load_documents().
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters repeated across adjacent chunks.

    Returns:
        List of smaller Document chunks ready for embedding.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Prefer natural boundaries before hard-cutting mid-word.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 4. Build vector store (embed + index)
# ---------------------------------------------------------------------------


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a cached-compatible HuggingFace embedding model instance.

    EMBEDDINGS
    ----------
    An embedding model maps text → a dense vector (here, 384 floats).
    Semantically similar text ends up with vectors that are close in space
    (high cosine similarity).

    BAAI/bge-small-en-v1.5:
      - Optimised for retrieval (query ↔ passage matching).
      - Runs locally via sentence-transformers (no API key required).
      - "small" variant balances speed and quality for hackathon / dev use.

    These vectors are what FAISS indexes and searches at query time.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine sim == dot product
    )


def build_vectorstore(
    chunks: list[Document],
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
    *,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Embed document chunks and build (or rebuild) a FAISS vector index.

    EMBEDDINGS + INDEXING
    ---------------------
    For each chunk:
      1. The embedding model encodes ``page_content`` into a 384-dim vector.
      2. FAISS stores vectors in an efficient index for fast similarity search.

    FAISS (Facebook AI Similarity Search):
      - In-memory index persisted to ``vectorstore_dir`` via save_local().
      - At query time we load the index and find the nearest chunk vectors.
      - Index is rebuilt when documents change (force_rebuild=True).

    Args:
        chunks: Chunked documents from split_documents().
        vectorstore_dir: Where the FAISS index and docstore are saved.
        force_rebuild: If True, delete any existing index before rebuilding.

    Returns:
        A FAISS vectorstore ready for similarity_search().
    """
    vectorstore_dir = Path(vectorstore_dir)

    if force_rebuild and vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)

    embeddings = _get_embeddings()

    if not chunks:
        # Return an empty store if called with no data (edge case guard).
        return FAISS.from_texts([""], embeddings, metadatas=[{"source": "empty"}])

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))

    return vectorstore


def load_vectorstore(
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
) -> Optional[FAISS]:
    """
    Load a previously saved FAISS index from disk.

    Returns None if no index exists yet.
    """
    vectorstore_dir = Path(vectorstore_dir)
    index_file = vectorstore_dir / "index.faiss"

    if not index_file.exists():
        return None

    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # local trusted index only
    )


# ---------------------------------------------------------------------------
# 5. Retrieve context
# ---------------------------------------------------------------------------


def retrieve_context(
    query: str,
    vectorstore: FAISS,
    top_k: int = TOP_K,
) -> list[Document]:
    """
    Retrieve the most relevant document chunks for a user query.

    RETRIEVAL
    ---------
    Steps:
      1. Prefix the query with the BGE instruction string so the embedding model
         treats it as a *search query* (not a passage to store).
      2. Embed the query → query vector.
      3. FAISS finds the ``top_k`` chunk vectors with highest cosine similarity.
      4. Return the matching Document objects (text + source metadata).

    Retrieved chunks become the "context" block injected into the LLM prompt.
    Poor retrieval → hallucinations or "I don't know" answers; good retrieval
    grounds the LLM in actual uploaded content.

    Args:
        query: The user's natural-language question.
        vectorstore: Loaded or freshly built FAISS index.
        top_k: Number of chunks to return.

    Returns:
        List of the top-k most relevant Document chunks.
    """
    if not query.strip():
        return []

    # BGE models perform better when queries use this retrieval prefix.
    query_for_embedding = BGE_QUERY_PREFIX + query

    results = vectorstore.similarity_search_with_score(query_for_embedding, k=top_k)

    # Drop weak matches so we don't cite passages that aren't actually relevant.
    return [doc for doc, distance in results if distance <= MAX_RETRIEVAL_DISTANCE]


# ---------------------------------------------------------------------------
# 6. Generate RAG answer
# ---------------------------------------------------------------------------


def _format_context(chunks: list[Document]) -> str:
    """Format retrieved chunks into a single context block for the LLM prompt."""
    if not chunks:
        return "No relevant context was found in the knowledge base."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        loc = f"{source}" + (f", page {page + 1}" if page is not None else "")
        parts.append(f"[{i}] ({loc})\n{chunk.page_content.strip()}")

    return "\n\n".join(parts)


def generate_rag_answer(
    query: str,
    context_chunks: list[Document],
    *,
    groq_api_key: Optional[str] = None,
    model: str = LLM_MODEL,
) -> str:
    """
    Generate a grounded answer using retrieved context and a Groq LLM.

    GENERATION
    ----------
    RAG generation = Retrieval + Augmented + Generation:
      - *Retrieval* already happened in retrieve_context().
      - *Augmented*: we inject retrieved chunks into the prompt as context.
      - *Generation*: the LLM reads context + question and writes an answer.

    The prompt instructs the model to:
      - Answer ONLY from the provided context.
      - Admit when context is insufficient (reduces hallucination).
      - Be concise and professional (enterprise assistant tone).

    Groq llama-3.1-8b-instant:
      - Fast inference via Groq's LPU hardware.
      - Large context window — enough for top-k chunks + question.
      - Requires GROQ_API_KEY in .env or environment (or passed explicitly).

    Args:
        query: User's question.
        context_chunks: Documents returned by retrieve_context().
        groq_api_key: Optional override; defaults to GROQ_API_KEY from .env / env.
        model: Groq model identifier.

    Returns:
        The LLM's text answer as a string.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required. Add it to .env (see .env.example) or pass "
            "groq_api_key= to generate_rag_answer()."
        )

    llm = ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.2,  # low temperature → factual, less creative drift
        max_tokens=1024,
    )

    context_text = _format_context(context_chunks)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an enterprise knowledge assistant. Answer ONLY from the context "
                "below — do not use outside knowledge. Paraphrase or quote closely from "
                "the passages; do not invent facts. When stating a fact, cite the source "
                "number like [1] or [2]. If the context does not contain the answer, reply: "
                "'I could not find this information in your uploaded documents.'\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": query})

    return response.content


# ---------------------------------------------------------------------------
# Convenience: full pipeline helpers (for app.py integration later)
# ---------------------------------------------------------------------------


def ingest_uploaded_files(
    uploaded_files: list[Any],
    *,
    upload_dir: Union[str, Path] = UPLOAD_DIR,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
    force_rebuild: bool = True,
) -> dict[str, Any]:
    """
    End-to-end ingestion: save → load → chunk → index.

    Intended for future use from app.py when the user uploads documents.

    Returns:
        Summary dict with paths, document count, chunk count, and vectorstore.
    """
    saved = save_uploaded_files(uploaded_files, upload_dir)
    docs = load_documents(upload_dir)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks, vectorstore_dir, force_rebuild=force_rebuild)

    return {
        "saved_paths": saved,
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "vectorstore": vectorstore,
    }


def answer_question(
    query: str,
    *,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
    top_k: int = TOP_K,
    groq_api_key: Optional[str] = None,
) -> dict[str, Any]:
    """
    End-to-end query: load index → retrieve → generate.

    Intended for future use from app.py when the user sends a chat message.

    Returns:
        Dict with ``answer``, ``context_chunks``, and ``sources`` metadata.
    """
    vectorstore = load_vectorstore(vectorstore_dir)
    if vectorstore is None:
        return {
            "answer": (
                "No knowledge base index found. Please upload and index "
                "documents before asking questions."
            ),
            "context_chunks": [],
            "sources": [],
        }

    chunks = retrieve_context(query, vectorstore, top_k=top_k)
    answer = generate_rag_answer(query, chunks, groq_api_key=groq_api_key)

    sources = [
        {
            "source": c.metadata.get("source", "unknown"),
            "page": c.metadata.get("page"),
            "preview": c.page_content[:200],
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "context_chunks": chunks,
        "sources": sources,
    }
