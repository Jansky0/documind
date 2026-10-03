import streamlit as st
from pathlib import Path
import time
from src.config import settings, UPLOADS_DIR
from src.document_loader import load_and_chunk_document
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DocuMind — RAG Document Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished portfolio appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .metric-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "api_key" not in st.session_state:
    st.session_state.api_key = settings.gemini_api_key

if "selected_model" not in st.session_state:
    st.session_state.selected_model = settings.gemini_model


# Vector Store & RAG Engine Cache
@st.cache_resource
def get_vector_store(api_key: str):
    return VectorStoreManager(api_key=api_key)


vector_store = get_vector_store(st.session_state.api_key)
rag_engine = RAGEngine(
    vector_store=vector_store,
    api_key=st.session_state.api_key,
    model_name=st.session_state.selected_model,
)

# ==========================================
# SIDEBAR: Configuration & Document Library
# ==========================================
with st.sidebar:
    st.markdown("### 🧠 **DocuMind**")
    st.caption("AI-Powered RAG Knowledge Base")
    st.divider()

    # API Key Configuration
    st.subheader("🔑 AI Configuration")
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get a free API key at https://aistudio.google.com/",
    )
    if user_api_key != st.session_state.api_key:
        st.session_state.api_key = user_api_key
        st.cache_resource.clear()
        st.rerun()

    # Model Selection
    model_options = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    current_model = (
        st.session_state.selected_model
        if st.session_state.selected_model in model_options
        else model_options[0]
    )
    selected_model = st.selectbox(
        "LLM Model",
        options=model_options,
        index=model_options.index(current_model),
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.rerun()

    top_k_val = st.slider("Top Chunks Retrieved (Top-K)", min_value=1, max_value=8, value=4)

    st.divider()

    # Document Upload Section
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF, Markdown, or TXT files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("⚡ Process & Index Documents", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_files = len(uploaded_files)
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                save_path = UPLOADS_DIR / uploaded_file.name

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load and chunk
                chunks = load_and_chunk_document(
                    save_path,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )

                # Store in Vector DB
                vector_store.add_chunks(chunks)
                progress_bar.progress((idx + 1) / total_files)

            status_text.empty()
            progress_bar.empty()
            st.success(f"Successfully indexed {total_files} file(s) into Vector Store!")
            time.sleep(1)
            st.rerun()

    st.divider()

    # Indexed Documents List
    st.subheader("📚 Indexed Documents")
    indexed_sources = vector_store.list_sources()

    if not indexed_sources:
        st.info("No documents indexed yet.")
    else:
        for item in indexed_sources:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 **{item['source']}**")
                st.caption(f"{item['chunks']} chunks • {item['page_count']} page(s)")
            with col2:
                if st.button("🗑️", key=f"del_{item['source']}", help="Delete this document"):
                    vector_store.delete_source(item["source"])
                    st.toast(f"Deleted: {item['source']}")
                    st.rerun()

        if st.button("🧹 Clear All Data", use_container_width=True, type="secondary"):
            vector_store.clear_all()
            st.session_state.chat_history = []
            st.success("All vector database records have been deleted.")
            st.rerun()

# ==========================================
# MAIN PANEL: Header, Metrics & Chat
# ==========================================
st.markdown('<div class="main-title">DocuMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Local-First Document Q&A Knowledge Base powered by RAG & Google Gemini with Grounded Citations</div>',
    unsafe_allow_html=True,
)

# Header Metrics
indexed_sources = vector_store.list_sources()
total_sources = len(indexed_sources)
total_chunks = sum(s["chunks"] for s in indexed_sources)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Indexed Documents", f"{total_sources} file(s)")
with col_m2:
    st.metric("Vector Chunks", f"{total_chunks} chunk(s)")
with col_m3:
    status_label = "🟢 Ready" if total_sources > 0 else "⚪ Awaiting Documents"
    st.metric("System Status", status_label)

st.divider()

# Welcome Hero if empty
if total_sources == 0:
    st.info(
        """
        👋 **Welcome to DocuMind!**
        
        To start querying your knowledge base:
        1. Enter your **Gemini API Key** in the left sidebar.
        2. Upload your documents (**PDF**, **Markdown**, or **TXT**).
        3. Click **"Process & Index Documents"**.
        4. Start asking questions in the chat box below!
        """
    )

# Render Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📚 Source Citations ({len(message['sources'])} snippet(s))"):
                for src in message["sources"]:
                    st.markdown(
                        f"""
                        **📄 `{src['source']}`** — *Page {src['page']}* (Relevance: `{src['score'] * 100:.1f}%`)  
                        > {src['text']}
                        """
                    )

# Chat Input & Streaming Execution
user_query = st.chat_input("Ask anything about your uploaded documents...")

if user_query:
    # Display User Message
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Prepare Assistant Response
    with st.chat_message("assistant"):
        # Stream response
        stream_generator = rag_engine.query_stream(user_query, top_k=top_k_val)
        response_text = st.write_stream(stream_generator)

        # Show Retrieved Citations
        retrieved_sources = rag_engine.last_retrieved_sources
        if retrieved_sources:
            with st.expander(f"📚 Source Citations ({len(retrieved_sources)} snippet(s))"):
                for src in retrieved_sources:
                    st.markdown(
                        f"""
                        **📄 `{src['source']}`** — *Page {src['page']}* (Relevance: `{src['score'] * 100:.1f}%`)  
                        > {src['text']}
                        """
                    )

    # Save to Session State
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": retrieved_sources,
        }
    )
