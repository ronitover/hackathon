import streamlit as st

st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#003d9b",
    "primary_container": "#0052cc",
    "on_primary": "#ffffff",
    "primary_fixed": "#dae2ff",
    "on_primary_fixed_variant": "#0040a2",
    "secondary_fixed": "#d9e2ff",
    "on_secondary_fixed_variant": "#00419d",
    "background": "#f8f9fb",
    "surface": "#f8f9fb",
    "surface_container_low": "#f3f4f6",
    "surface_container": "#edeef0",
    "surface_container_high": "#e7e8ea",
    "surface_container_lowest": "#ffffff",
    "on_surface": "#191c1e",
    "on_surface_variant": "#434654",
    "outline_variant": "#c3c6d6",
    "outline": "#737685",
    "error": "#ba1a1a",
    "error_container": "#ffdad6",
}

PROFILE_IMG = (
    "https://lh3.googleusercontent.com/aida-public/AB6AXuB6VyasX9Z2O0z098HbUZs2_"
    "wkmDFTzmd-6qY_p2_q_zRcu7C5Z9zJAtum5FsvMsRGIjtGvayN28HsJLqeYOg226AC3bWkmb"
    "Vx9TCXdduYEid3sKsbV_HpAPs39MYvOvFRRE0Q-mvn7f-bufifnPMCoIt_dNyeqpe9GfxzBk"
    "PbVK1OpL_VFI-tbHxf6JsEGF76Dzj7aCv9aqQb9bZs002h3PGXtKXqsC-v-44SI4gY1-x3wn"
    "VrhapbiWqV_jhlPBeKWhYShvTs46TlF"
)

SUGGESTIONS = [
    ("summarize", "Summarize the uploaded document", "Get a concise executive summary of key points and takeaways."),
    ("policy", "What policies are mentioned?", "Extract specific compliance, HR, or operational guidelines automatically."),
    ("hail", "Explain the onboarding process", "Understand internal workflows and employee lifecycle procedures."),
    ("target", "What are the key objectives?", "Identify strategic goals and mission-critical targets in the data."),
]

def init_state():
    defaults = {
        "messages": [],
        "prompt_area": "",
        "documents": [],
        "show_welcome": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def doc_count() -> int:
    return len(st.session_state.documents)


def chunk_count() -> int:
    return sum(max(1, len(doc.get("name", "")) // 8) for doc in st.session_state.documents)


def kb_active() -> bool:
    return doc_count() > 0


def inject_styles():
    c = COLORS
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=JetBrains+Mono&display=swap" rel="stylesheet"/>
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
        <style>
            html, body, [class*="css"] {{
                font-family: 'Inter', sans-serif !important;
            }}

            #MainMenu, footer, header[data-testid="stHeader"] {{
                visibility: hidden !important;
                height: 0 !important;
            }}

            .stDeployButton, [data-testid="stToolbar"], [data-testid="stStatusWidget"] {{
                display: none !important;
            }}

            .block-container {{
                padding: 0 !important;
                max-width: 100% !important;
            }}

            [data-testid="stAppViewContainer"] > .main {{
                background: {c["background"]};
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #f3f4f6 0%, #edeef0 100%) !important;
                border-right: 1px solid {c["outline_variant"]};
                min-width: 260px !important;
                width: 260px !important;
            }}

            [data-testid="stSidebar"] > div:first-child {{
                padding: 1.5rem 1rem 1rem;
            }}

            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                gap: 0.4rem !important;
            }}

            .sidebar-action {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem;
                border-radius: 0.5rem;
                font-size: 14px;
                color: {c["on_surface_variant"]};
                cursor: pointer;
                margin: 0.1rem 0;
            }}

            .sidebar-action:hover {{
                background: {c["surface_container_high"]};
            }}

            .sidebar-action.danger {{
                color: {c["error"]};
            }}

            .sidebar-action.danger:hover {{
                background: {c["error_container"]};
            }}

            .sidebar-clear-click {{
                margin-top: -40px !important;
                margin-bottom: 0 !important;
                position: relative;
                z-index: 2;
            }}

            .sidebar-clear-click .stButton > button {{
                height: 40px !important;
                width: 100% !important;
                opacity: 0 !important;
                cursor: pointer !important;
                border: none !important;
                padding: 0 !important;
            }}

            [data-testid="stFileUploader"] {{
                padding: 0;
                margin-bottom: 1.25rem;
            }}

            [data-testid="stFileUploader"] section {{
                padding: 0;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
                background: {c["primary"]};
                border: none;
                border-radius: 0.5rem;
                padding: 0.75rem 1rem;
                min-height: 44px;
                box-shadow: 0 4px 12px rgba(9, 30, 66, 0.08);
                cursor: pointer;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]::before {{
                content: "upload_file";
                font-family: 'Material Symbols Outlined';
                font-size: 18px;
                color: {c["on_primary"]};
                margin-right: 8px;
                vertical-align: middle;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]::after {{
                content: "Document Upload";
                color: {c["on_primary"]};
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.05em;
                vertical-align: middle;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div {{
                display: none !important;
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]:hover {{
                background: {c["primary_container"]};
            }}

            [data-testid="stFileUploader"] [data-testid="stFileUploadDropzoneInstructions"],
            [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {{
                font-size: 11px;
            }}

            .material-symbols-outlined {{
                font-family: 'Material Symbols Outlined' !important;
                font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
                vertical-align: middle;
                line-height: 1;
            }}

            .ms-fill {{
                font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            }}

            .pulse-dot {{
                animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }}

            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.4; }}
            }}

            /* Main layout */
            section.main .block-container {{
                display: flex;
                flex-direction: column;
                min-height: 100vh;
                padding-top: 0 !important;
            }}

            .topbar {{
                height: 64px;
                border-bottom: 1px solid {c["outline_variant"]};
                background: {c["surface"]};
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 1.5rem;
                flex-shrink: 0;
                width: 100%;
                box-sizing: border-box;
            }}

            .topbar-left {{
                display: flex;
                align-items: center;
                gap: 2rem;
            }}

            .topbar-title {{
                font-size: 24px;
                font-weight: 800;
                color: {c["on_surface"]};
                letter-spacing: -0.01em;
                white-space: nowrap;
            }}

            .topbar-right {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }}

            .status-pill {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.05em;
                white-space: nowrap;
            }}

            .pill-rag {{
                background: {c["primary_fixed"]};
                color: {c["on_primary_fixed_variant"]};
            }}

            .pill-memory {{
                background: {c["secondary_fixed"]};
                color: {c["on_secondary_fixed_variant"]};
            }}

            .avatar {{
                width: 32px;
                height: 32px;
                border-radius: 50%;
                border: 1px solid {c["outline"]};
                object-fit: cover;
            }}

            .content-scroll {{
                flex: 1;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2rem 1.5rem 1rem;
                width: 100%;
                box-sizing: border-box;
            }}

            .content-inner {{
                width: 100%;
                max-width: 48rem;
            }}

            .chat-dock-marker {{
                display: none;
            }}

            .element-container:has(.chat-dock-marker),
            .element-container:has(.chat-dock-marker) ~ .element-container:has([data-testid="stTextArea"]),
            .element-container:has(.chat-toolbar-block),
            .element-container:has(.chat-footer-note) {{
                max-width: 56rem;
                margin-left: auto !important;
                margin-right: auto !important;
                padding-left: 1.5rem !important;
                padding-right: 1.5rem !important;
                width: 100%;
                box-sizing: border-box;
            }}

            .element-container:has(.chat-footer-note) {{
                padding-bottom: 2rem !important;
            }}

            /* Hero */
            .hero-card {{
                background: white;
                border: 1px solid {c["outline_variant"]};
                border-radius: 0.5rem;
                padding: 2rem;
                box-shadow: 0 4px 12px rgba(9, 30, 66, 0.08);
                position: relative;
                overflow: hidden;
                margin-bottom: 1.5rem;
            }}

            .hero-glow {{
                position: absolute;
                right: -48px;
                top: -48px;
                width: 256px;
                height: 256px;
                background: rgba(0, 61, 155, 0.05);
                border-radius: 50%;
                filter: blur(48px);
                pointer-events: none;
            }}

            .hero-icon {{
                background: {c["primary_fixed"]};
                color: {c["on_primary_fixed_variant"]};
                width: 48px;
                height: 48px;
                border-radius: 0.5rem;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.5rem;
            }}

            .hero-title {{
                font-size: 30px;
                font-weight: 700;
                line-height: 38px;
                letter-spacing: -0.02em;
                color: {c["on_surface"]};
                margin: 0 0 1rem 0;
            }}

            .hero-subtitle {{
                font-size: 16px;
                line-height: 24px;
                color: {c["on_surface_variant"]};
                max-width: 36rem;
                margin: 0 0 2rem 0;
            }}

            .action-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }}

            .action-card {{
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem;
                background: {c["surface_container_low"]};
                border: 1px solid {c["outline_variant"]};
                border-radius: 0.5rem;
            }}

            .action-card-icon {{
                background: white;
                padding: 0.75rem;
                border-radius: 0.5rem;
                border: 1px solid {c["outline_variant"]};
                color: {c["on_surface_variant"]};
                flex-shrink: 0;
            }}

            .action-card-title {{
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.05em;
                color: {c["on_surface"]};
                margin: 0;
            }}

            .action-card-desc {{
                font-size: 12px;
                color: {c["on_surface_variant"]};
                margin: 0;
            }}

            /* Suggestion cards */
            .suggest-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
            }}

            .suggestion-card {{
                background: white;
                border: 1px solid {c["outline_variant"]};
                border-radius: 0.5rem;
                padding: 1.5rem;
                cursor: pointer;
                transition: border-color 0.15s, box-shadow 0.15s;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                min-height: 130px;
            }}

            .suggestion-card:hover {{
                border-color: {c["primary_container"]};
                box-shadow: 0 4px 12px rgba(9, 30, 66, 0.08);
            }}

            .suggestion-head {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 0.5rem;
            }}

            .suggestion-icon {{
                color: {c["primary"]};
                font-size: 22px;
            }}

            .suggestion-arrow {{
                color: {c["on_surface_variant"]};
                font-size: 16px;
            }}

            .suggestion-title {{
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.05em;
                color: {c["on_surface"]};
                margin: 0;
            }}

            .suggestion-desc {{
                font-size: 12px;
                color: {c["on_surface_variant"]};
                margin: 0.25rem 0 0 0;
                line-height: 1.4;
            }}

            .suggest-slot {{
                height: 0;
                overflow: visible;
                margin: 0;
                padding: 0;
            }}

            .suggest-click {{
                margin-top: -132px !important;
                margin-bottom: 1rem !important;
                position: relative;
                z-index: 2;
            }}

            .suggest-click .stButton > button {{
                height: 132px !important;
                width: 100% !important;
                opacity: 0 !important;
                cursor: pointer !important;
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
            }}

            /* Chat input */
            .element-container:has(.chat-dock-marker) ~ .element-container [data-testid="stTextArea"] {{
                margin-bottom: 0 !important;
                background: white;
                border: 2px solid {c["outline_variant"]};
                border-radius: 1rem 1rem 0 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }}

            .element-container:has(.chat-dock-marker) ~ .element-container [data-testid="stTextArea"] textarea {{
                border: none !important;
                box-shadow: none !important;
                background: transparent !important;
                font-size: 14px !important;
                min-height: 100px !important;
                padding: 1.5rem !important;
                resize: none !important;
            }}

            .chat-toolbar-block {{
                background: rgba(237, 238, 240, 0.5);
            }}

            .element-container:has(.chat-toolbar-block) {{
                background: white;
                border: 2px solid {c["outline_variant"]};
                border-top: 1px solid {c["outline_variant"]};
                border-radius: 0 0 1rem 1rem;
                margin-top: -1rem !important;
                padding: 0.25rem 0.5rem 0.5rem !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }}

            .element-container:has(.chat-toolbar-block) [data-testid="stHorizontalBlock"] {{
                align-items: center !important;
            }}

            .toolbar-left {{
                display: flex;
                align-items: center;
                gap: 1rem;
                color: {c["on_surface_variant"]};
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
            }}

            .toolbar-divider {{
                width: 1px;
                height: 24px;
                background: {c["outline_variant"]};
            }}

            .send-btn .stButton > button {{
                background: {c["primary"]} !important;
                color: {c["on_primary"]} !important;
                border-radius: 0.5rem !important;
                font-size: 12px !important;
                font-weight: 600 !important;
                letter-spacing: 0.05em !important;
                padding: 0.5rem 1.25rem !important;
                border: none !important;
                white-space: nowrap !important;
            }}

            .send-btn .stButton > button::after {{
                content: "send";
                font-family: 'Material Symbols Outlined';
                font-size: 18px;
                margin-left: 6px;
                vertical-align: middle;
            }}

            .send-btn .stButton > button:hover {{
                background: {c["primary_container"]} !important;
            }}

            .send-btn .stButton {{
                margin: 0 !important;
            }}

            .chat-footer-note {{
                text-align: center;
                font-size: 12px;
                color: {c["on_surface_variant"]};
                opacity: 0.6;
                margin-top: 1rem;
            }}

            /* Messages */
            .msg-user {{
                background: {c["primary_fixed"]};
                color: {c["on_primary_fixed_variant"]};
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                font-size: 14px;
            }}

            .msg-assistant {{
                background: white;
                border: 1px solid {c["outline_variant"]};
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
                font-size: 14px;
            }}

            /* Sidebar pieces */
            .kb-empty {{
                padding: 2rem 1rem;
                border: 2px dashed {c["outline_variant"]};
                border-radius: 0.5rem;
                text-align: center;
                opacity: 0.6;
                margin: 0.25rem 0 1rem;
            }}

            .kb-doc-item {{
                font-size: 12px;
                color: {c["on_surface_variant"]};
                padding: 0.35rem 0.5rem;
                background: {c["surface_container_lowest"]};
                border: 1px solid {c["outline_variant"]};
                border-radius: 0.25rem;
                margin-bottom: 0.25rem;
            }}

            .status-card {{
                background: {c["surface_container"]};
                border: 1px solid {c["outline_variant"]};
                border-radius: 0.5rem;
                padding: 1rem;
                margin: 0.75rem 0;
            }}

            .stat-box {{
                text-align: center;
                padding: 0.5rem;
                background: {c["surface_container_lowest"]};
                border: 1px solid {c["outline_variant"]};
                border-radius: 0.5rem;
            }}

            .stat-label {{
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.05em;
                color: {c["on_surface_variant"]};
                margin: 0;
            }}

            .stat-value {{
                font-size: 18px;
                font-weight: 600;
                color: {c["primary"]};
                margin: 0;
            }}

            .sidebar-spacer {{
                flex: 1;
                min-height: 2rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand():
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;">
            <div style="background:{COLORS['primary']};padding:8px;border-radius:0.5rem;display:flex;">
                <span class="material-symbols-outlined ms-fill" style="color:{COLORS['on_primary']};font-size:20px;">database</span>
            </div>
            <div>
                <div style="font-size:18px;font-weight:700;color:{COLORS['primary']};line-height:1.2;">Enterprise RAG</div>
                <div style="font-size:12px;color:{COLORS['on_surface_variant']};">Intelligence Control</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_kb():
    count = doc_count()
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:0 0.5rem;margin-bottom:0.5rem;">
            <span style="font-size:12px;font-weight:600;letter-spacing:0.05em;color:{COLORS['on_surface_variant']};text-transform:uppercase;">Knowledge Base</span>
            <span style="background:{COLORS['surface_container_high']};padding:2px 8px;border-radius:999px;font-family:'JetBrains Mono',monospace;font-size:13px;">{count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if count == 0:
        st.markdown(
            f"""
            <div class="kb-empty">
                <span class="material-symbols-outlined" style="font-size:28px;color:{COLORS['on_surface_variant']};">folder_open</span>
                <p style="font-size:12px;color:{COLORS['on_surface_variant']};margin:0.5rem 0 0 0;">No documents indexed yet.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for doc in st.session_state.documents:
            st.markdown(
                f'<div class="kb-doc-item"><span class="material-symbols-outlined" style="font-size:14px;">description</span> {doc["name"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid {COLORS['outline_variant']};">
            <div style="display:flex;align-items:center;gap:0.5rem;padding:0.5rem;color:{COLORS['on_surface_variant']};">
                <span class="material-symbols-outlined" style="font-size:20px;">monitoring</span>
                <span style="font-size:14px;">Analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status():
    active = kb_active()
    status_color = "#16a34a" if active else COLORS["error"]
    status_label = "Active" if active else "Inactive"
    pulse_class = "" if active else "pulse-dot"

    st.markdown(
        f"""
        <div class="status-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                <span style="font-size:12px;font-weight:600;letter-spacing:0.05em;color:{COLORS['on_surface_variant']};">KB STATUS</span>
                <div style="display:flex;align-items:center;gap:4px;">
                    <div class="{pulse_class}" style="width:8px;height:8px;border-radius:50%;background:{status_color};"></div>
                    <span style="font-size:10px;font-weight:600;color:{status_color};text-transform:uppercase;">{status_label}</span>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div class="stat-box">
                    <p class="stat-label">DOCS</p>
                    <p class="stat-value">{doc_count()}</p>
                </div>
                <div class="stat-box">
                    <p class="stat-label">CHUNKS</p>
                    <p class="stat-value">{chunk_count()}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_topbar():
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-left">
                <span class="topbar-title">RAG Assistant</span>
            </div>
            <div class="topbar-right">
                <div class="status-pill pill-rag">
                    <span class="material-symbols-outlined ms-fill" style="font-size:14px;">security</span>
                    RAG ENABLED
                </div>
                <div class="status-pill pill-memory">
                    <span class="material-symbols-outlined ms-fill" style="font-size:14px;">memory</span>
                    MEMORY ENABLED
                </div>
                <div style="width:1px;height:32px;background:{COLORS['outline_variant']};"></div>
                <span class="material-symbols-outlined" style="color:{COLORS['on_surface_variant']};font-size:22px;">notifications</span>
                <img class="avatar" src="{PROFILE_IMG}" alt="User profile"/>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-glow"></div>
            <div style="position:relative;z-index:1;">
                <div class="hero-icon">
                    <span class="material-symbols-outlined" style="font-size:32px;">neurology</span>
                </div>
                <h2 class="hero-title">Welcome to your AI <br/><span style="color:{COLORS['primary']};">Knowledge Assistant</span></h2>
                <p class="hero-subtitle">
                    Upload documents to create your enterprise knowledge assistant. Our RAG engine will index your content to provide grounded, traceable, and secure answers.
                </p>
                <div class="action-grid">
                    <div class="action-card">
                        <div class="action-card-icon">
                            <span class="material-symbols-outlined">upload</span>
                        </div>
                        <div>
                            <p class="action-card-title">Start Indexing</p>
                            <p class="action-card-desc">PDF, DOCX, or TXT files</p>
                        </div>
                    </div>
                    <div class="action-card">
                        <div class="action-card-icon">
                            <span class="material-symbols-outlined">menu_book</span>
                        </div>
                        <div>
                            <p class="action-card-title">Explore Docs</p>
                            <p class="action-card-desc">View knowledge base</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_suggestion_card(icon: str, title: str, desc: str, idx: int):
    st.markdown(
        f"""
        <div class="suggestion-card">
            <div class="suggestion-head">
                <span class="material-symbols-outlined suggestion-icon">{icon}</span>
                <span class="material-symbols-outlined suggestion-arrow">north_east</span>
            </div>
            <p class="suggestion-title">{title}</p>
            <p class="suggestion-desc">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="suggest-click">', unsafe_allow_html=True)
    if st.button(" ", key=f"suggest_{idx}"):
        st.session_state.prompt_area = title
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_suggestions():
    row1 = st.columns(2, gap="medium")
    row2 = st.columns(2, gap="medium")
    slots = [row1[0], row1[1], row2[0], row2[1]]
    for idx, (icon, title, desc) in enumerate(SUGGESTIONS):
        with slots[idx]:
            _render_suggestion_card(icon, title, desc, idx)


def render_messages():
    for msg in st.session_state.messages:
        css = "msg-user" if msg["role"] == "user" else "msg-assistant"
        st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)


def handle_send(prompt: str):
    prompt = prompt.strip()
    if not prompt:
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.show_welcome = False
    if kb_active():
        reply = (
            f"I found relevant context across {doc_count()} document(s). "
            f"Here's a grounded response to: *{prompt}*"
        )
    else:
        reply = (
            "No documents are indexed yet. Upload PDF, DOCX, or TXT files "
            "via **Document Upload** in the sidebar to enable RAG answers."
        )
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.prompt_area = ""


def render_chat_input():
    prompt = st.text_area(
        "chat",
        placeholder="Ask anything about your enterprise knowledge...",
        label_visibility="collapsed",
        height=100,
        key="prompt_area",
    )
    st.markdown('<div class="chat-toolbar-block">', unsafe_allow_html=True)
    tool_left, tool_right = st.columns([3, 1])
    with tool_left:
        st.markdown(
            f"""
            <div class="toolbar-left">
                <span class="material-symbols-outlined" style="font-size:20px;">attach_file</span>
                <span class="material-symbols-outlined" style="font-size:20px;">mic</span>
                <div class="toolbar-divider"></div>
                <span>Markdown Supported</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tool_right:
        st.markdown('<div class="send-btn">', unsafe_allow_html=True)
        if st.button("Send Request", key="send_btn"):
            handle_send(prompt)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<p class="chat-footer-note chat-footer-note">AI can make mistakes. Check important info. Secure & Private Enterprise Instance.</p>',
        unsafe_allow_html=True,
    )


def process_uploads(uploaded_files):
    if not uploaded_files:
        return
    existing = {d["name"] for d in st.session_state.documents}
    for f in uploaded_files:
        if f.name not in existing:
            st.session_state.documents.append({"name": f.name, "size": f.size})
    st.session_state.show_welcome = len(st.session_state.messages) == 0


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
init_state()

inject_styles()

with st.sidebar:
    render_sidebar_brand()

    uploaded = st.file_uploader(
        "Document Upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    process_uploads(uploaded)

    render_sidebar_kb()

    st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

    render_sidebar_status()

    st.markdown(
        f"""
        <div class="sidebar-action">
            <span class="material-symbols-outlined" style="font-size:20px;">settings</span>
            Settings
        </div>
        <div class="sidebar-action danger" id="clear-chat-action">
            <span class="material-symbols-outlined" style="font-size:20px;">delete_sweep</span>
            Clear Chat
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-clear-click">', unsafe_allow_html=True)
    if st.button(" ", key="clear_btn"):
        st.session_state.messages = []
        st.session_state.prompt_area = ""
        st.session_state.show_welcome = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

render_topbar()

st.markdown('<div class="content-scroll"><div class="content-inner">', unsafe_allow_html=True)

if st.session_state.messages:
    render_messages()
elif st.session_state.show_welcome:
    render_hero()
    render_suggestions()

st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown('<div class="chat-dock-marker"></div>', unsafe_allow_html=True)
render_chat_input()
