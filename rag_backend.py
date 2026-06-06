"""
RAG backend for the Enterprise RAG Assistant.

This module implements the full retrieval-augmented generation pipeline:
  upload → parse → chunk → embed → index → retrieve → generate
"""

from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"
MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.1-8b-instant"

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_RETRIEVAL_DISTANCE = 1.15
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# How many prior user/assistant turns to include in memory-aware RAG.
MEMORY_TURN_LIMIT = 12

# Follow-up cues that benefit from query rewriting using chat history.
FOLLOWUP_CUES = (
    "it", "its", "they", "them", "their", "that", "this", "those", "these",
    "who", "what about", "how about", "tell me more", "explain more",
    "and ", "also ", "same", "above", "previous",
)

NO_INFO_DECLINE_PHRASES = (
    "could not find",
    "cannot find",
    "can't find",
    "do not have information",
    "don't have information",
    "not find this information",
    "not contain enough",
    "no relevant passages",
    "i could not find this",
)


# ---------------------------------------------------------------------------
# Manifest helpers (track indexed files for skip / incremental updates)
# ---------------------------------------------------------------------------


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def get_upload_manifest(upload_dir: Union[str, Path] = UPLOAD_DIR) -> dict[str, dict[str, int]]:
    upload_dir = Path(upload_dir)
    if not upload_dir.exists():
        return {}

    manifest: dict[str, dict[str, int]] = {}
    for file_path in sorted(upload_dir.iterdir()):
        if file_path.is_file() and not file_path.name.startswith("."):
            manifest[file_path.name] = _file_signature(file_path)
    return manifest


def _read_index_manifest() -> Optional[dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_index_manifest(
    files: dict[str, dict[str, int]],
    chunk_count: int,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
) -> None:
    vectorstore_dir = Path(vectorstore_dir)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps({"files": files, "chunk_count": chunk_count}, indent=2),
        encoding="utf-8",
    )


def vectorstore_index_mtime(vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR) -> float:
    index_file = Path(vectorstore_dir) / "index.faiss"
    if not index_file.exists():
        return 0.0
    return index_file.stat().st_mtime


def vectorstore_exists(vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR) -> bool:
    return (Path(vectorstore_dir) / "index.faiss").exists()


# ---------------------------------------------------------------------------
# 1. Save / delete uploaded files
# ---------------------------------------------------------------------------


def save_uploaded_files(
    uploaded_files: list[Any],
    upload_dir: Union[str, Path] = UPLOAD_DIR,
) -> list[Path]:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for uploaded in uploaded_files:
        dest = upload_dir / uploaded.name
        if hasattr(uploaded, "getbuffer"):
            data = uploaded.getbuffer()
        else:
            uploaded.seek(0)
            data = uploaded.read()
        dest.write_bytes(data)
        saved_paths.append(dest)

    return saved_paths


def delete_uploaded_file(
    filename: str,
    upload_dir: Union[str, Path] = UPLOAD_DIR,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
) -> bool:
    """Remove a file from uploads and invalidate the vector index."""
    file_path = Path(upload_dir) / filename
    if not file_path.exists():
        return False

    file_path.unlink()
    _clear_vectorstore(vectorstore_dir)
    return True


def _clear_vectorstore(vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR) -> None:
    vectorstore_dir = Path(vectorstore_dir)
    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)


# ---------------------------------------------------------------------------
# 2. Load & parse documents
# ---------------------------------------------------------------------------


def _load_single_file(file_path: Path) -> list[Document]:
    loader_cls = LOADER_MAP.get(file_path.suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return loader_cls(str(file_path)).load()


def load_documents(
    upload_dir: Union[str, Path] = UPLOAD_DIR,
    *,
    filenames: Optional[list[str]] = None,
    skip_unsupported: bool = True,
) -> tuple[list[Document], list[str]]:
    """
    Load supported documents from the upload directory.

    Returns:
        (documents, warnings) — warnings list skipped/unsupported filenames.
    """
    upload_dir = Path(upload_dir)
    if not upload_dir.exists():
        return [], []

    documents: list[Document] = []
    warnings: list[str] = []

    paths = sorted(upload_dir.iterdir())
    if filenames is not None:
        paths = [upload_dir / name for name in filenames]

    for file_path in paths:
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        suffix = file_path.suffix.lower()
        if suffix not in LOADER_MAP:
            message = f"Skipped unsupported file '{file_path.name}' (supported: {', '.join(LOADER_MAP)})"
            if skip_unsupported:
                warnings.append(message)
                continue
            raise ValueError(message)

        try:
            documents.extend(_load_single_file(file_path))
        except Exception as exc:
            warnings.append(f"Failed to load '{file_path.name}': {exc}")

    return documents, warnings


# ---------------------------------------------------------------------------
# 3. Chunk documents
# ---------------------------------------------------------------------------


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 4. Build vector store (embed + index)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    chunks: list[Document],
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
    *,
    force_rebuild: bool = False,
) -> FAISS:
    vectorstore_dir = Path(vectorstore_dir)

    if force_rebuild and vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir)

    embeddings = _get_embeddings()

    if not chunks:
        return FAISS.from_texts([""], embeddings, metadatas=[{"source": "empty"}])

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    return vectorstore


def load_vectorstore(
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
) -> Optional[FAISS]:
    vectorstore_dir = Path(vectorstore_dir)
    if not (vectorstore_dir / "index.faiss").exists():
        return None

    return FAISS.load_local(
        str(vectorstore_dir),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _manifest_diff(
    current: dict[str, dict[str, int]],
    indexed: dict[str, dict[str, int]],
) -> tuple[set[str], set[str], set[str]]:
    current_names = set(current)
    indexed_names = set(indexed)
    removed = indexed_names - current_names
    added = current_names - indexed_names
    changed = {
        name
        for name in current_names & indexed_names
        if current[name] != indexed[name]
    }
    return added, removed, changed


def build_knowledge_base(
    upload_dir: Union[str, Path] = UPLOAD_DIR,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
) -> dict[str, Any]:
    """
    Index all uploaded files. Skips rebuild if manifest unchanged; merges new
    files incrementally when possible; full rebuild on removals or edits.
    """
    upload_dir = Path(upload_dir)
    current_manifest = get_upload_manifest(upload_dir)
    stored_manifest = _read_index_manifest()
    indexed_files = (stored_manifest or {}).get("files", {})

    if not current_manifest:
        return {
            "ok": False,
            "error": "Upload at least one supported document (PDF, DOCX, or TXT) first.",
            "warnings": [],
            "chunk_count": 0,
            "document_count": 0,
            "skipped_rebuild": False,
        }

    index_ready = vectorstore_exists(vectorstore_dir)

    if (
        index_ready
        and stored_manifest
        and current_manifest == indexed_files
    ):
        return {
            "ok": True,
            "warnings": [],
            "chunk_count": stored_manifest.get("chunk_count", 0),
            "document_count": len(current_manifest),
            "skipped_rebuild": True,
            "vectorstore": load_vectorstore(vectorstore_dir),
        }

    added, removed, changed = _manifest_diff(current_manifest, indexed_files)

    if index_ready and stored_manifest and not removed and not changed and added:
        docs, warnings = load_documents(upload_dir, filenames=sorted(added))
        new_chunks = split_documents(docs)
        if not new_chunks:
            return {
                "ok": False,
                "error": "No text could be extracted from the new file(s).",
                "warnings": warnings,
                "chunk_count": stored_manifest.get("chunk_count", 0),
                "document_count": len(current_manifest),
                "skipped_rebuild": False,
            }

        vectorstore = load_vectorstore(vectorstore_dir)
        if vectorstore is None:
            return build_knowledge_base(upload_dir, vectorstore_dir)

        new_store = FAISS.from_documents(new_chunks, _get_embeddings())
        vectorstore.merge_from(new_store)
        vectorstore_dir = Path(vectorstore_dir)
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(vectorstore_dir))

        chunk_count = vectorstore.index.ntotal
        _write_index_manifest(current_manifest, chunk_count, vectorstore_dir)

        return {
            "ok": True,
            "warnings": warnings,
            "chunk_count": chunk_count,
            "document_count": len(current_manifest),
            "skipped_rebuild": False,
            "incremental": True,
            "vectorstore": vectorstore,
        }

    docs, warnings = load_documents(upload_dir)
    if not docs:
        return {
            "ok": False,
            "error": "No readable content found in uploaded files.",
            "warnings": warnings,
            "chunk_count": 0,
            "document_count": 0,
            "skipped_rebuild": False,
        }

    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks, vectorstore_dir, force_rebuild=True)
    _write_index_manifest(current_manifest, len(chunks), vectorstore_dir)

    return {
        "ok": True,
        "warnings": warnings,
        "chunk_count": len(chunks),
        "document_count": len(current_manifest),
        "skipped_rebuild": False,
        "vectorstore": vectorstore,
    }


def get_indexed_chunk_count(vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR) -> int:
    repair_manifest_if_missing(vectorstore_dir=vectorstore_dir)
    stored = _read_index_manifest()
    if stored and "chunk_count" in stored:
        return int(stored["chunk_count"])
    vectorstore = load_vectorstore(vectorstore_dir)
    if vectorstore is None:
        return 0
    return vectorstore.index.ntotal


def repair_manifest_if_missing(
    upload_dir: Union[str, Path] = UPLOAD_DIR,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
) -> bool:
    """Write manifest.json for a legacy index that predates manifest tracking."""
    if MANIFEST_PATH.exists():
        return False

    index_file = Path(vectorstore_dir) / "index.faiss"
    if not index_file.exists():
        return False

    import faiss

    index = faiss.read_index(str(index_file))
    current_manifest = get_upload_manifest(upload_dir)
    if not current_manifest:
        return False

    _write_index_manifest(current_manifest, index.ntotal, vectorstore_dir)
    return True


# ---------------------------------------------------------------------------
# 5. Retrieve context
# ---------------------------------------------------------------------------


def retrieve_context(
    query: str,
    vectorstore: FAISS,
    top_k: int = TOP_K,
    *,
    chat_history: Optional[list[dict]] = None,
) -> list[Document]:
    if not query.strip():
        return []

    search_query = build_retrieval_query(query, chat_history)
    query_for_embedding = BGE_QUERY_PREFIX + search_query
    results = vectorstore.similarity_search_with_score(query_for_embedding, k=top_k)
    return [doc for doc, distance in results if distance <= MAX_RETRIEVAL_DISTANCE]


def _looks_like_followup(query: str) -> bool:
    lower = query.lower().strip()
    if len(lower.split()) <= 8:
        return True
    return any(cue in lower for cue in FOLLOWUP_CUES)


def build_retrieval_query(query: str, chat_history: Optional[list[dict]] = None) -> str:
    """
    Rewrite short or pronoun-heavy follow-ups using recent conversation so
    vector search retrieves relevant passages.
    """
    if not chat_history:
        return query

    if not _looks_like_followup(query):
        return query

    recent_user = [
        msg["content"].strip()
        for msg in chat_history
        if msg.get("role") == "user" and msg.get("content", "").strip()
    ]
    recent_assistant = [
        msg["content"].strip()
        for msg in chat_history
        if msg.get("role") == "assistant" and msg.get("content", "").strip()
    ]

    parts: list[str] = []
    if recent_user:
        parts.extend(recent_user[-2:])
    if recent_assistant:
        # Last assistant reply helps disambiguate "they" / "those roles".
        parts.append(recent_assistant[-1][:200])

    if not parts:
        return query

    return f"{' '.join(parts)} {query}".strip()


def _to_langchain_messages(chat_history: list[dict], *, limit: int = MEMORY_TURN_LIMIT) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for msg in chat_history[-limit:]:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# ---------------------------------------------------------------------------
# 6. Generate RAG answer
# ---------------------------------------------------------------------------


def _format_context(chunks: list[Document]) -> str:
    if not chunks:
        return "No relevant context was found in the knowledge base."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        loc = f"{source}" + (f", page {page + 1}" if page is not None else "")
        parts.append(f"[{i}] ({loc})\n{chunk.page_content.strip()}")

    return "\n\n".join(parts)


def format_sources_for_ui(chunks: list[Document], *, excerpt_len: int = 280) -> list[dict]:
    sources: list[dict] = []
    for chunk in chunks:
        excerpt = chunk.page_content.strip().replace("\n", " ")
        if len(excerpt) > excerpt_len:
            excerpt = excerpt[:excerpt_len] + "…"
        sources.append(
            {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page"),
                "excerpt": excerpt,
            }
        )
    return sources


def answer_indicates_no_info(reply: str) -> bool:
    lower = reply.lower()
    return any(phrase in lower for phrase in NO_INFO_DECLINE_PHRASES)


def generate_rag_answer(
    query: str,
    context_chunks: list[Document],
    *,
    chat_history: Optional[list[dict]] = None,
    groq_api_key: Optional[str] = None,
    model: str = LLM_MODEL,
) -> str:
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required. Add it to .env (see .env.example) or pass "
            "groq_api_key= to generate_rag_answer()."
        )

    llm = ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.2,
        max_tokens=1024,
    )

    history_messages = _to_langchain_messages(chat_history or [])
    prompt_messages = [
        (
            "system",
            "You are an enterprise knowledge assistant. Answer ONLY from the context "
            "below — do not use outside knowledge. Use the conversation history to "
            "understand follow-up questions (e.g. 'it', 'they', 'how many days'), but "
            "ground every fact in the document context, not in prior answers alone. "
            "Paraphrase or quote closely from the passages; do not invent facts. When "
            "stating a fact, cite the source number like [1] or [2]. If the context does "
            "not contain the answer, reply: 'I could not find this information in your "
            "uploaded documents.'\n\n"
            "Context:\n{context}",
        ),
    ]
    if history_messages:
        prompt_messages.append(MessagesPlaceholder("chat_history"))
    prompt_messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt | llm

    invoke_args: dict[str, Any] = {
        "context": _format_context(context_chunks),
        "question": query,
    }
    if history_messages:
        invoke_args["chat_history"] = history_messages

    response = chain.invoke(invoke_args)
    return response.content


def generate_rag_answer_stream(
    query: str,
    context_chunks: list[Document],
    *,
    chat_history: Optional[list[dict]] = None,
    groq_api_key: Optional[str] = None,
    model: str = LLM_MODEL,
):
    """Stream tokens from the RAG answer generator."""
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is required. Add it to .env (see .env.example) or pass "
            "groq_api_key= to generate_rag_answer_stream()."
        )

    llm = ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.2,
        max_tokens=1024,
        streaming=True,
    )

    history_messages = _to_langchain_messages(chat_history or [])
    prompt_messages = [
        (
            "system",
            "You are an enterprise knowledge assistant. Answer ONLY from the context "
            "below — do not use outside knowledge. Use the conversation history to "
            "understand follow-up questions (e.g. 'it', 'they', 'how many days'), but "
            "ground every fact in the document context, not in prior answers alone. "
            "Paraphrase or quote closely from the passages; do not invent facts. When "
            "stating a fact, cite the source number like [1] or [2]. If the context does "
            "not contain the answer, reply: 'I could not find this information in your "
            "uploaded documents.'\n\n"
            "Context:\n{context}",
        ),
    ]
    if history_messages:
        prompt_messages.append(MessagesPlaceholder("chat_history"))
    prompt_messages.append(("human", "{question}"))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt | llm

    invoke_args: dict[str, Any] = {
        "context": _format_context(context_chunks),
        "question": query,
    }
    if history_messages:
        invoke_args["chat_history"] = history_messages

    for chunk in chain.stream(invoke_args):
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        if text:
            yield text


# ---------------------------------------------------------------------------
# Pipeline helpers (used by app.py)
# ---------------------------------------------------------------------------


def ingest_uploaded_files(
    uploaded_files: list[Any],
    *,
    upload_dir: Union[str, Path] = UPLOAD_DIR,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
) -> dict[str, Any]:
    saved = save_uploaded_files(uploaded_files, upload_dir)
    result = build_knowledge_base(upload_dir, vectorstore_dir)
    result["saved_paths"] = saved
    return result


def answer_question(
    query: str,
    vectorstore: Optional[FAISS] = None,
    *,
    chat_history: Optional[list[dict]] = None,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
    top_k: int = TOP_K,
    groq_api_key: Optional[str] = None,
) -> dict[str, Any]:
    if vectorstore is None:
        vectorstore = load_vectorstore(vectorstore_dir)

    if vectorstore is None:
        return {
            "answer": (
                "No knowledge base index found. Please upload and index "
                "documents before asking questions."
            ),
            "context_chunks": [],
            "sources": [],
            "grounded": False,
            "no_index": True,
        }

    chunks = retrieve_context(query, vectorstore, top_k=top_k, chat_history=chat_history)
    if not chunks:
        return {
            "answer": "I could not find relevant passages in your uploaded documents for this question.",
            "context_chunks": [],
            "sources": [],
            "grounded": False,
            "no_index": False,
        }

    answer = generate_rag_answer(
        query,
        chunks,
        chat_history=chat_history,
        groq_api_key=groq_api_key,
    )
    grounded = not answer_indicates_no_info(answer)

    return {
        "answer": answer,
        "context_chunks": chunks,
        "sources": format_sources_for_ui(chunks) if grounded else [],
        "grounded": grounded,
        "no_index": False,
        "retrieval_query": build_retrieval_query(query, chat_history),
    }


def answer_question_stream(
    query: str,
    vectorstore: Optional[FAISS] = None,
    *,
    chat_history: Optional[list[dict]] = None,
    vectorstore_dir: Union[str, Path] = VECTORSTORE_DIR,
    top_k: int = TOP_K,
    groq_api_key: Optional[str] = None,
):
    """
    Like answer_question but yields (event, payload) tuples for streaming UI.

    Events: retrieval_done, token, complete, error
    """
    if vectorstore is None:
        vectorstore = load_vectorstore(vectorstore_dir)

    if vectorstore is None:
        yield ("error", {"answer": "No knowledge base index found. Please upload and index documents first."})
        return

    chunks = retrieve_context(query, vectorstore, top_k=top_k, chat_history=chat_history)
    yield (
        "retrieval_done",
        {
            "context_chunks": chunks,
            "retrieval_query": build_retrieval_query(query, chat_history),
            "history_len": len(chat_history or []),
        },
    )

    if not chunks:
        yield (
            "complete",
            {
                "answer": "I could not find relevant passages in your uploaded documents for this question.",
                "context_chunks": [],
                "sources": [],
                "grounded": False,
            },
        )
        return

    tokens: list[str] = []
    for token in generate_rag_answer_stream(
        query,
        chunks,
        chat_history=chat_history,
        groq_api_key=groq_api_key,
    ):
        tokens.append(token)
        yield ("token", token)

    answer = "".join(tokens)
    grounded = not answer_indicates_no_info(answer)
    yield (
        "complete",
        {
            "answer": answer,
            "context_chunks": chunks,
            "sources": format_sources_for_ui(chunks) if grounded else [],
            "grounded": grounded,
        },
    )
