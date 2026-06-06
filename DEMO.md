# Live demo script — Enterprise RAG Assistant

Use this flow when presenting to judges (~3 minutes).

## Before you start

1. `streamlit run app.py`
2. Confirm sidebar shows **RAG MODE** and knowledge base is **Ready**
3. If not indexed: upload `docs/sample-documents/HR-Roles-and-Responsibilities-PDF.pdf` → **Build Knowledge Base**

## Demo conversation (shows RAG + memory)

### Turn 1 — baseline retrieval
**You:** What HR roles are defined in the document?

**Expected:** Answer listing roles from the PDF with `[1]` citations and source expander.

---

### Turn 2 — pronoun follow-up (memory + retrieval rewrite)
**You:** What are their main responsibilities?

**Expected:**
- Sidebar **Chat history** shows Turn 1
- Status: “Using N prior message(s) from session memory”
- Optional: “Memory-aware search” with expanded query
- Answer about responsibilities, grounded in same document

---

### Turn 3 — short follow-up
**You:** Which role handles onboarding?

**Expected:** Understands “which role” in context of prior HR discussion.

---

### Turn 4 — prove persistence (optional wow moment)
1. Copy the URL from the browser (contains `?session=...`)
2. Refresh the page
3. **Chat history and messages should reload**

---

## One-liner for judges

> “We scoped to **RAG mode only** per guidance. The assistant retrieves from uploaded enterprise documents, cites sources, and uses **SQLite session memory** so follow-up questions work — including after a browser refresh via the session URL.”

## Scope note

| Implemented | Intentionally omitted |
|-------------|----------------------|
| RAG document Q&A | General Chat (Mode 1) |
| Source citations | Web Search (Mode 2) |
| SQLite short-term memory | LangGraph multi-agent routing |
| Streaming responses | Multi-user auth |
