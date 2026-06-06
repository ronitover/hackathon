import html
from pathlib import Path

import streamlit as st

import memory
from rag_backend import (
    UPLOAD_DIR,
    answer_question_stream,
    build_knowledge_base,
    delete_uploaded_file,
    get_indexed_chunk_count,
    get_upload_manifest,
    repair_manifest_if_missing,
    save_uploaded_files,
    vectorstore_exists,
    vectorstore_index_mtime,
)
from rag_backend import load_vectorstore as _load_vectorstore

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PDF_ICON_SVG = (
    '<svg class="file-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" '
    'stroke="#A32D2D" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
    '<polyline points="14 2 14 8 20 8"/></svg>'
)


@st.cache_resource
def get_cached_vectorstore(index_mtime: float):
    if index_mtime <= 0:
        return None
    return _load_vectorstore()


def get_vectorstore():
    return get_cached_vectorstore(vectorstore_index_mtime())


def invalidate_vectorstore_cache():
    get_cached_vectorstore.clear()


def resolve_session_id() -> str:
    """Persist session in URL so refresh keeps chat history."""
    param = st.query_params.get("session")
    if isinstance(param, list):
        param = param[0] if param else None

    if "session_id" in st.session_state:
        sid = st.session_state.session_id
        if param != sid:
            st.query_params["session"] = sid
        return sid

    if param:
        st.session_state.session_id = param
        memory.ensure_session(param)
        return param

    sid = memory.new_session_id()
    st.session_state.session_id = sid
    st.query_params["session"] = sid
    memory.ensure_session(sid)
    return sid


def load_messages_from_db(session_id: str) -> list[dict]:
    return memory.get_messages(session_id)


def init_state():
    memory.init_db()
    session_id = resolve_session_id()

    defaults = {
        "documents": [],
        "show_welcome": True,
        "chunk_count": 0,
        "kb_indexed": False,
        "is_generating": False,
        "ui_rag_mode": True,
        "ui_memory_on": True,
        "pending_delete": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "messages" not in st.session_state:
        st.session_state.messages = load_messages_from_db(session_id)
        if st.session_state.messages:
            st.session_state.show_welcome = False

    if vectorstore_exists():
        repair_manifest_if_missing()
        st.session_state.kb_indexed = True
        st.session_state.chunk_count = get_indexed_chunk_count()

    sync_documents_from_disk()


def sync_documents_from_disk():
    if not UPLOAD_DIR.exists():
        st.session_state.documents = []
        return

    st.session_state.documents = [
        {"name": f.name, "size": f.stat().st_size}
        for f in sorted(UPLOAD_DIR.iterdir())
        if f.is_file() and not f.name.startswith(".")
    ]


def doc_count() -> int:
    return len(st.session_state.documents)


def kb_active() -> bool:
    return st.session_state.kb_indexed and vectorstore_exists()


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

            :root {
                --color-brand: #185FA5;
                --color-danger-file: #A32D2D;
                --color-background-primary: #ffffff;
                --color-background-secondary: #f8fafc;
                --color-text-primary: #0f172a;
                --color-text-secondary: #475569;
                --color-text-tertiary: #94a3b8;
                --color-border-secondary: #e2e8f0;
                --color-border-tertiary: #f1f5f9;
                --sidebar-width: 22rem;
            }

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif !important;
            }

            #MainMenu, footer {
                visibility: hidden !important;
                height: 0 !important;
            }

            .stDeployButton, [data-testid="stToolbar"], [data-testid="stStatusWidget"] {
                display: none !important;
            }

            header[data-testid="stHeader"] {
                visibility: visible !important;
                height: 3rem !important;
                background: transparent !important;
            }

            section[data-testid="stSidebar"],
            [data-testid="stSidebar"] {
                background: var(--color-background-secondary) !important;
                border-right: 0.5px solid var(--color-border-secondary) !important;
                transform: translateX(0px) !important;
                visibility: visible !important;
                display: block !important;
                min-width: var(--sidebar-width) !important;
                width: var(--sidebar-width) !important;
            }

            [data-testid="stSidebar"] > div:first-child {
                height: 100vh;
                overflow-y: auto;
                padding: 0.85rem 1rem 1rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.5rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
                padding: 0 !important;
            }

            [data-testid="stSidebar"] [data-testid="element-container"] {
                margin-bottom: 0 !important;
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
                margin: 0 !important;
            }

            [data-testid="stSidebar"] hr,
            [data-testid="stSidebar"] [data-testid="stDivider"] {
                margin: 0.75rem 0 !important;
                border-color: var(--color-border-secondary) !important;
            }

            .sb-sidebar-footer {
                margin: 0.75rem 0 0.5rem 0;
                border: none;
                border-top: 0.5px solid var(--color-border-secondary);
                height: 0;
            }

            [data-testid="stSidebar"] .stButton > button {
                border-radius: 8px !important;
                font-size: 0.8125rem !important;
                font-weight: 500 !important;
                min-height: 2.25rem !important;
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: var(--color-brand) !important;
                border: 1px solid var(--color-brand) !important;
                color: #fff !important;
            }

            [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                background: var(--color-background-primary) !important;
                border: 0.5px solid var(--color-border-secondary) !important;
                color: var(--color-text-primary) !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] {
                background: var(--color-background-primary);
                border: 1px dashed var(--color-border-secondary);
                border-radius: 10px;
                padding: 0.55rem 0.65rem;
                margin-bottom: 0.25rem;
            }

            [data-testid="stSidebar"] [data-testid="stAlert"] {
                font-size: 0.78rem !important;
                padding: 0.55rem 0.75rem !important;
                border-radius: 8px !important;
            }

            .sb-brand {
                display: flex;
                align-items: center;
                gap: 0.65rem;
                margin-bottom: 0.65rem;
            }

            .sb-brand-icon {
                width: 2.25rem;
                height: 2.25rem;
                border-radius: 10px;
                background: var(--color-brand);
                color: #fff;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1rem;
                font-weight: 700;
                flex-shrink: 0;
            }

            .sb-brand-title {
                font-size: 0.95rem;
                font-weight: 600;
                color: var(--color-text-primary);
                margin: 0;
                line-height: 1.25;
            }

            .sb-brand-sub {
                font-size: 0.72rem;
                color: var(--color-text-secondary);
                margin: 0;
                line-height: 1.3;
            }

            .toggle-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.35rem 0;
            }

            [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
            [data-testid="stSidebar"] [data-testid="stToggle"] label p {
                font-size: 0.8125rem !important;
                font-weight: 500 !important;
                color: var(--color-text-primary) !important;
            }

            [data-testid="stSidebar"] [data-testid="stToggle"] {
                margin-bottom: 0.25rem;
            }

            [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.file-row) {
                margin-bottom: 0.5rem !important;
                align-items: center !important;
            }

            [data-testid="stSidebar"] [data-testid="stToggle"] label {
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                width: 100% !important;
                gap: 0.75rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stToggle"] [data-testid="stToggleSwitch"] {
                width: 22px !important;
                height: 12px !important;
                min-width: 22px !important;
                flex-shrink: 0 !important;
            }

            [data-testid="stSidebar"] [data-testid="stToggle"] [data-testid="stToggleSwitch"] > div {
                width: 22px !important;
                height: 12px !important;
            }

            [data-testid="stSidebar"] [data-testid="stToggle"] [data-testid="stToggleSwitch"][aria-checked="true"] {
                background: #185FA5 !important;
            }

            [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.icon-btn-trash) > [data-testid="column"]:last-child {
                flex: 0 0 1.75rem !important;
                width: 1.75rem !important;
                min-width: 1.75rem !important;
                max-width: 1.75rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.icon-btn-trash) > [data-testid="column"]:first-child {
                min-width: 0 !important;
            }

            [data-testid="stSidebar"] .icon-btn-trash .stButton {
                width: 1.75rem !important;
            }

            [data-testid="stSidebar"] .icon-btn-trash .stButton > button {
                min-width: 1.75rem !important;
                width: 1.75rem !important;
                min-height: 1.75rem !important;
                height: 1.75rem !important;
                padding: 0 !important;
                font-size: 0 !important;
                color: transparent !important;
                border: none !important;
                background: transparent center/14px 14px no-repeat !important;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") !important;
                box-shadow: none !important;
            }

            [data-testid="stSidebar"] .icon-btn-trash .stButton > button:hover {
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%23475569' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") !important;
            }

            .sb-section-label {
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: var(--color-text-tertiary);
                margin: 0 0 0.45rem 0;
                display: block;
                clear: both;
            }

            .sb-section-label.sb-section-divider {
                margin-top: 0.75rem;
                padding-top: 0.75rem;
                border-top: 0.5px solid var(--color-border-secondary);
            }

            .sb-section-label.sb-section-first {
                margin-top: 0.75rem;
            }

            .file-row {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                background: var(--color-background-primary);
                border: 0.5px solid var(--color-border-secondary);
                border-radius: 8px;
                padding: 0.5rem 0.65rem;
                margin-bottom: 0.35rem;
            }

            .file-icon {
                flex-shrink: 0;
            }

            .file-info {
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                gap: 0.1rem;
            }

            .file-name {
                font-size: 0.78rem;
                font-weight: 500;
                color: var(--color-text-primary);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .file-meta {
                font-size: 0.68rem;
                color: var(--color-text-tertiary);
            }

            .delete-confirm {
                font-size: 0.72rem;
                color: var(--color-text-secondary);
                background: var(--color-background-primary);
                border: 0.5px solid var(--color-border-secondary);
                border-radius: 8px;
                padding: 0.45rem 0.65rem;
                margin-bottom: 0.35rem;
            }

            .delete-confirm-actions .stButton > button {
                min-height: 1.75rem !important;
                font-size: 0.72rem !important;
            }

            .sb-empty {
                font-size: 0.78rem;
                color: var(--color-text-tertiary);
                background: var(--color-background-primary);
                border: 1px dashed var(--color-border-secondary);
                border-radius: 10px;
                padding: 0.85rem 0.75rem;
                text-align: center;
                line-height: 1.45;
            }

            .history-panel {
                background: var(--color-background-primary);
                border: 0.5px solid var(--color-border-secondary);
                border-radius: 10px;
                overflow-y: auto;
                max-height: 12rem;
            }

            .history-item {
                font-size: 0.76rem;
                color: var(--color-text-secondary);
                padding: 0.6rem 0.75rem;
                border-bottom: 0.5px solid var(--color-border-tertiary);
                line-height: 1.4;
            }

            .history-item:last-child { border-bottom: none; }

            .history-q {
                display: block;
                color: var(--color-text-primary);
                font-weight: 500;
                font-size: 0.78rem;
                margin-bottom: 0.2rem;
            }

            .history-tag {
                display: inline-block;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                padding: 0.1rem 0.35rem;
                border-radius: 4px;
                margin-bottom: 0.25rem;
            }

            .history-tag-grounded {
                background: var(--color-border-tertiary);
                color: var(--color-brand);
            }

            .history-tag-general {
                background: var(--color-border-tertiary);
                color: var(--color-text-tertiary);
            }

            .history-preview {
                color: var(--color-text-tertiary);
                font-size: 0.72rem;
            }

            .welcome-shell {
                text-align: center;
                padding: 3.5rem 1rem 2rem;
                max-width: 28rem;
                margin: 0 auto;
            }

            .welcome-icon {
                width: 44px;
                height: 44px;
                border-radius: 12px;
                background: var(--color-border-tertiary);
                color: var(--color-brand);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1rem;
            }

            .welcome-shell h2 {
                font-size: 17px;
                font-weight: 500;
                color: var(--color-text-primary);
                margin: 0 0 0.35rem 0;
            }

            .welcome-sub {
                font-size: 13px;
                color: var(--color-text-secondary);
                margin: 0 0 1.75rem 0;
                line-height: 1.45;
            }

            .steps-row {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                flex-wrap: wrap;
            }

            .step {
                display: flex;
                align-items: center;
                gap: 0.35rem;
                font-size: 12px;
                color: var(--color-text-tertiary);
            }

            .step-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                border: 1.5px solid var(--color-border-secondary);
                background: var(--color-background-primary);
                flex-shrink: 0;
            }

            .step.done {
                color: var(--color-text-secondary);
            }

            .step.done .step-dot {
                background: var(--color-brand);
                border-color: var(--color-brand);
            }

            .step.active {
                color: var(--color-brand);
                font-weight: 500;
            }

            .step.active .step-dot {
                background: var(--color-brand);
                border-color: var(--color-brand);
                box-shadow: 0 0 0 3px var(--color-border-tertiary);
            }

            .step-arrow {
                color: var(--color-text-tertiary);
                font-size: 11px;
            }

            [data-testid="stChatMessage"] {
                background: transparent !important;
                padding: 0.75rem 0 !important;
            }

            [data-testid="stChatMessageAvatarUser"] {
                background: var(--color-brand) !important;
            }

            [data-testid="stChatMessageAvatarAssistant"] {
                background: var(--color-border-tertiary) !important;
                color: var(--color-brand) !important;
            }

            .grounded-note {
                font-size: 0.75rem;
                color: var(--color-text-tertiary);
                margin-top: 0.25rem;
                font-style: italic;
            }

            .source-card {
                background: var(--color-background-secondary);
                border: 0.5px solid var(--color-border-secondary);
                border-left: 3px solid var(--color-brand);
                border-radius: 8px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
            }

            .source-card-title {
                font-size: 0.85rem;
                font-weight: 600;
                color: var(--color-text-primary);
                margin: 0 0 0.35rem 0;
            }

            .source-card-excerpt {
                font-size: 0.8rem;
                color: var(--color-text-secondary);
                margin: 0;
                line-height: 1.45;
            }

            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stSidebarCollapseButton"] {
                display: none !important;
            }

            [data-testid="stAppViewContainer"] > .main {
                background: var(--color-background-primary);
                display: flex;
                flex-direction: column;
                min-height: 100vh;
            }

            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 0 !important;
                max-width: 48rem !important;
                margin: 0 auto !important;
                flex: 1;
            }

            [data-testid="stBottomBlockContainer"],
            [data-testid="stBottom"] {
                border-top: 0.5px solid var(--color-border-secondary) !important;
                background: var(--color-background-primary) !important;
                padding: 0.75rem 0 1rem !important;
                max-width: 48rem !important;
                margin: 0 auto !important;
            }

            [data-testid="stChatInput"] {
                background: transparent !important;
            }

            [data-testid="stChatInput"] > div {
                background: var(--color-background-secondary) !important;
                border: 0.5px solid var(--color-border-tertiary) !important;
                border-radius: 12px !important;
                padding: 0.15rem 0.35rem !important;
            }

            [data-testid="stChatInput"] textarea {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0.65rem 0.75rem !important;
                font-size: 0.875rem !important;
            }

            [data-testid="stChatInput"] button {
                background: var(--color-brand) !important;
                border-radius: 8px !important;
                width: 2rem !important;
                height: 2rem !important;
                min-height: 2rem !important;
            }

            [data-testid="stChatInput"] button svg {
                display: none !important;
            }

            [data-testid="stChatInput"] button::after {
                content: "↑";
                color: #fff;
                font-size: 1rem;
                font-weight: 600;
                line-height: 1;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _file_subline(doc: dict) -> str:
    size = _format_bytes(doc["size"])
    if kb_active() and st.session_state.chunk_count > 0 and doc_count() == 1:
        return f"{size} · {st.session_state.chunk_count} chunks"
    return size


def render_sidebar_brand():
    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-brand-icon">R</div>
            <div>
                <p class="sb-brand-title">Enterprise RAG</p>
                <p class="sb-brand-sub">Knowledge Assistant</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.toggle("RAG Mode", key="ui_rag_mode")
    st.toggle("Memory On", key="ui_memory_on")


def render_documents_list():
    chunks = st.session_state.chunk_count if kb_active() else 0
    label = "Indexed Files"
    if chunks > 0 and doc_count() != 1:
        label += f" · {chunks} chunks"
    st.markdown(f'<p class="sb-section-label sb-section-first">{label}</p>', unsafe_allow_html=True)

    if not st.session_state.documents:
        st.markdown(
            '<div class="sb-empty">No documents uploaded.<br>Add PDF, DOCX, or TXT below.</div>',
            unsafe_allow_html=True,
        )
        return

    for doc in st.session_state.documents:
        fname = doc["name"]
        if st.session_state.pending_delete == fname:
            st.markdown('<div class="delete-confirm">Remove this file?</div>', unsafe_allow_html=True)
            yes_col, no_col = st.columns(2)
            with yes_col:
                st.markdown('<div class="delete-confirm-actions">', unsafe_allow_html=True)
                if st.button("Yes", key=f"del_yes_{fname}", use_container_width=True):
                    st.session_state.pending_delete = None
                    remove_document(fname)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with no_col:
                st.markdown('<div class="delete-confirm-actions">', unsafe_allow_html=True)
                if st.button("Cancel", key=f"del_no_{fname}", use_container_width=True):
                    st.session_state.pending_delete = None
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            continue

        row_col, trash_col = st.columns([12, 1], vertical_alignment="center", gap="small")
        meta = _file_subline(doc)
        with row_col:
            st.markdown(
                f"""
                <div class="file-row">
                    {PDF_ICON_SVG}
                    <div class="file-info">
                        <span class="file-name">{html.escape(fname)}</span>
                        <span class="file-meta">{meta}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with trash_col:
            st.markdown('<div class="icon-btn-trash">', unsafe_allow_html=True)
            if st.button(" ", key=f"del_{fname}", help=f"Remove {fname}"):
                st.session_state.pending_delete = fname
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_source_citations(sources: list[dict]):
    if not sources:
        return

    with st.expander(
        f"📎 {len(sources)} source passage{'s' if len(sources) != 1 else ''} from your documents",
        expanded=False,
    ):
        for i, source in enumerate(sources, start=1):
            filename = html.escape(Path(source.get("source", "unknown")).name)
            page = source.get("page")
            page_label = f" · Page {page + 1}" if page is not None else ""
            excerpt = html.escape(source.get("excerpt", ""))
            st.markdown(
                f"""
                <div class="source-card">
                    <p class="source-card-title">[{i}] {filename}{page_label}</p>
                    <p class="source-card-excerpt">"{excerpt}"</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("grounded") and msg.get("sources"):
                st.markdown(
                    '<p class="grounded-note">Synthesized from retrieved document passages — expand sources to see exact excerpts.</p>',
                    unsafe_allow_html=True,
                )
                render_source_citations(msg["sources"])


def render_welcome_empty():
    has_docs = doc_count() > 0
    indexed = kb_active()
    subtext = (
        "Your knowledge base is indexed and ready."
        if indexed
        else "Upload documents and build your index to begin."
    )

    if indexed:
        s1, s2, s3 = "step done", "step done", "step active"
    elif has_docs:
        s1, s2, s3 = "step done", "step active", "step"
    else:
        s1, s2, s3 = "step active", "step", "step"

    st.markdown(
        f"""
        <div class="welcome-shell">
            <div class="welcome-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
            </div>
            <h2>Ask your documents anything</h2>
            <p class="welcome-sub">{html.escape(subtext)}</p>
            <div class="steps-row">
                <div class="{s1}"><span class="step-dot"></span>Upload docs</div>
                <span class="step-arrow">→</span>
                <div class="{s2}"><span class="step-dot"></span>Build index</div>
                <span class="step-arrow">→</span>
                <div class="{s3}"><span class="step-dot"></span>Ask anything</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_panel(session_id: str):
    stats = memory.get_session_stats(session_id)
    count = stats["message_count"]
    st.markdown(
        f'<p class="sb-section-label sb-section-divider">Chat History · {count} message{"s" if count != 1 else ""}</p>',
        unsafe_allow_html=True,
    )

    summaries = memory.get_history_summaries(session_id)
    if not summaries:
        st.markdown(
            '<div class="sb-empty">No conversation yet.<br>Ask a question to begin.</div>',
            unsafe_allow_html=True,
        )
        return

    items_html = ""
    for item in reversed(summaries):
        tag_class = "history-tag-grounded" if item.get("grounded") else "history-tag-general"
        tag_text = "Cited" if item.get("grounded") else "Response"
        q = html.escape(item["question"][:72] + ("…" if len(item["question"]) > 72 else ""))
        preview = html.escape(item["answer_preview"])
        items_html += (
            f'<div class="history-item">'
            f'<span class="history-tag {tag_class}">{tag_text}</span>'
            f'<span class="history-q">{q}</span>'
            f'<span class="history-preview">{preview}</span>'
            f"</div>"
        )
    st.markdown(f'<div class="history-panel">{items_html}</div>', unsafe_allow_html=True)


def generate_pending_answer():
    """Stream answer once, persist full message + sources to SQLite."""
    query = st.session_state.messages[-1]["content"]
    session_id = st.session_state.session_id
    chat_history = memory.get_history_for_rag(session_id)[:-1]

    result = {"content": "", "sources": [], "grounded": False}

    with st.chat_message("assistant"):
        if not kb_active():
            result["content"] = (
                "Upload documents in the sidebar, then click **Build Knowledge Base** "
                "before asking questions."
            )
            st.markdown(result["content"])
        else:
            try:
                vectorstore = get_vectorstore()
                if chat_history:
                    st.caption(f"🧠 Using {len(chat_history)} prior message(s) from session memory")

                def token_generator():
                    nonlocal result
                    context_chunks = []
                    for event, payload in answer_question_stream(
                        query,
                        vectorstore=vectorstore,
                        chat_history=chat_history,
                    ):
                        if event == "retrieval_done":
                            context_chunks = payload.get("context_chunks", [])
                            rq = payload.get("retrieval_query", query)
                            if rq != query:
                                short = rq[:90] + "…" if len(rq) > 90 else rq
                                st.caption(f"🔍 Memory-aware search: _{short}_")
                            if not context_chunks:
                                result = {
                                    "content": "I could not find relevant passages in your uploaded documents for this question.",
                                    "sources": [],
                                    "grounded": False,
                                }
                                yield result["content"]
                                return
                        elif event == "token":
                            result["content"] += payload
                            yield payload
                        elif event == "complete":
                            result = {
                                "content": payload["answer"],
                                "sources": payload.get("sources", []),
                                "grounded": payload.get("grounded", False),
                            }
                        elif event == "error":
                            result = {"content": payload["answer"], "sources": [], "grounded": False}
                            yield payload["answer"]
                            return

                streamed = st.write_stream(token_generator())
                if streamed and not result["content"]:
                    result["content"] = streamed if isinstance(streamed, str) else "".join(streamed)

                if result.get("grounded") and result.get("sources"):
                    st.markdown(
                        '<p class="grounded-note">Synthesized from retrieved document passages.</p>',
                        unsafe_allow_html=True,
                    )
                    render_source_citations(result["sources"])

            except ValueError as exc:
                result = {"content": str(exc), "sources": [], "grounded": False}
                st.error(str(exc))
            except Exception as exc:
                result = {"content": f"Failed to generate answer: {exc}", "sources": [], "grounded": False}
                st.error(result["content"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["content"],
            "sources": result.get("sources", []),
            "grounded": result.get("grounded", False),
        }
    )
    memory.add_message(
        session_id,
        "assistant",
        result["content"],
        grounded=result.get("grounded", False),
        sources=result.get("sources"),
    )
    st.session_state.is_generating = False


def run_build_knowledge_base():
    with st.status("Building knowledge base…", expanded=True) as status:
        status.write("📂 Loading uploaded files…")
        result = build_knowledge_base()

        for warning in result.get("warnings", []):
            status.write(f"⚠️ {warning}")

        if not result["ok"]:
            st.session_state.build_kb_message = ("error", result.get("error", "Indexing failed."))
            status.update(label="Indexing failed", state="error")
            return

        if result.get("skipped_rebuild"):
            status.write("✅ Index is already up to date — skipped rebuild.")
        elif result.get("incremental"):
            status.write(f"➕ Added new file(s) to existing index ({result['chunk_count']} chunks total).")
        else:
            status.write(f"✂️ Indexed {result['document_count']} file(s) into {result['chunk_count']} chunks.")
            status.write("🧠 Embeddings complete.")

        status.update(label="Knowledge base ready", state="complete")

    invalidate_vectorstore_cache()
    st.session_state.chunk_count = result["chunk_count"]
    st.session_state.kb_indexed = True

    if result.get("skipped_rebuild"):
        msg = f"Index up to date · {result['chunk_count']} chunks."
    elif result.get("incremental"):
        msg = f"Added new file(s) · {result['chunk_count']} chunks total."
    else:
        msg = f"Indexed {doc_count()} file(s) · {result['chunk_count']} chunks."

    st.session_state.build_kb_message = ("success", msg)


def _uploads_changed(uploaded_files) -> bool:
    on_disk = get_upload_manifest()
    uploaded_meta = {f.name: {"size": f.size, "mtime_ns": 0} for f in uploaded_files}
    on_disk_sizes = {name: meta["size"] for name, meta in on_disk.items()}

    if set(uploaded_meta) != set(on_disk_sizes):
        return True
    return any(uploaded_meta[name]["size"] != on_disk_sizes[name] for name in uploaded_meta)


def process_uploads(uploaded_files):
    if uploaded_files:
        if _uploads_changed(uploaded_files):
            save_uploaded_files(uploaded_files)
            invalidate_vectorstore_cache()
            st.session_state.kb_indexed = False
            st.session_state.chunk_count = 0
            st.session_state.upload_notice = "New files saved — click **Build Knowledge Base** to index them."
        else:
            save_uploaded_files(uploaded_files)

    sync_documents_from_disk()
    st.session_state.show_welcome = len(st.session_state.messages) == 0


def remove_document(filename: str):
    if delete_uploaded_file(filename):
        invalidate_vectorstore_cache()
        st.session_state.kb_indexed = False
        st.session_state.chunk_count = 0
        sync_documents_from_disk()
        st.session_state.upload_notice = f"Removed **{filename}** — rebuild the knowledge base to update search."


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
init_state()
inject_styles()

with st.sidebar:
    render_sidebar_brand()

    render_documents_list()

    st.markdown('<p class="sb-section-label sb-section-divider">Upload</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    process_uploads(uploaded)

    if st.session_state.get("upload_notice"):
        st.info(st.session_state.upload_notice)
        del st.session_state.upload_notice

    if st.button("Build Knowledge Base", key="build_kb", use_container_width=True, type="primary"):
        run_build_knowledge_base()
        st.rerun()

    if "build_kb_message" in st.session_state:
        level, text = st.session_state.build_kb_message
        if level == "success":
            st.success(text)
        else:
            st.error(text)
        del st.session_state.build_kb_message

    render_history_panel(st.session_state.session_id)

    st.markdown('<hr class="sb-sidebar-footer">', unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True, type="secondary"):
        memory.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.show_welcome = True
        st.session_state.is_generating = False
        st.rerun()

if st.session_state.messages:
    render_messages()
elif st.session_state.show_welcome:
    render_welcome_empty()

if st.session_state.is_generating:
    generate_pending_answer()
    st.rerun()

if prompt := st.chat_input("Ask anything about your documents…", disabled=st.session_state.is_generating):
    st.session_state.messages.append({"role": "user", "content": prompt})
    memory.add_message(st.session_state.session_id, "user", prompt)
    st.session_state.show_welcome = False
    st.session_state.is_generating = True
    st.rerun()
