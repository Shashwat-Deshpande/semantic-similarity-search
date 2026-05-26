"""
Semantic Similarity Search — Production-Ready Streamlit App
Uses ChromaDB (in-memory) + Sentence-Transformers (all-MiniLM-L6-v2)
"""

import streamlit as st

# ──────────────────────────────────────────────
# 1. PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 2. GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    .result-doc { font-size: 1rem; color: #1e293b; margin-bottom: .35rem; }

    .result-meta { font-size: 0.8rem; color: #64748b; }

    .badge {
        display: inline-block;
        padding: .2rem .6rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 600;
        background: #ede9fe;
        color: #4f46e5;
    }

    .stat-box {
        background: #f1f5f9;
        border-radius: 10px;
        padding: .75rem 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .stat-num { font-size: 1.6rem; font-weight: 700; color: #4f46e5; }
    .stat-lbl { font-size: .75rem; color: #64748b; text-transform: uppercase; }

    hr { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 3. LOAD RESOURCES
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_resources():
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.EphemeralClient()
        hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        return client, hf_ef, None

    except ImportError as exc:
        return None, None, f"Missing dependency: {exc}"
    except Exception as exc:
        return None, None, f"Init error: {exc}"


client, hf_ef, load_error = load_resources()

if load_error:
    st.error(f"⚠️ {load_error}")
    st.stop()

# ──────────────────────────────────────────────
# 4. CONSTANTS
# ──────────────────────────────────────────────
COLLECTION_NAME = "hf_similarity_search"

SAMPLE_DOCS = """\
The quick brown fox jumps over the lazy dog.
I am incredibly happy and full of joy because I topped my university exams.
The planetary alignment of Mars and Jupiter will be visible tonight.
Quantum computing leverages superposition.
A Mediterranean diet reduces cardiovascular risk.
Machine learning requires large datasets.
The Amazon rainforest produces oxygen and hosts biodiversity.\
"""

# ──────────────────────────────────────────────
# 5. SESSION STATE
# ──────────────────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state["indexed"] = False
if "doc_count" not in st.session_state:
    st.session_state["doc_count"] = 0

# ──────────────────────────────────────────────
# 6. SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Document Ingestion")

    raw_text = st.text_area(
        "Documents",
        value=SAMPLE_DOCS,
        height=260,
        label_visibility="collapsed",
    )

    index_btn = st.button("⚡ Index Documents", type="primary")

    if index_btn:
        try:
            try:
                client.delete_collection(name=COLLECTION_NAME)
            except:
                pass

            collection = client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=hf_ef,
            )

            # Using splitlines handles multi-paragraph line breaks properly across all machines
            documents = [x.strip() for x in raw_text.splitlines() if x.strip()]

            if not documents:
                st.error("No documents found.")
            else:
                ids = [f"doc_{i}" for i in range(len(documents))]

                with st.spinner("Embedding documents…"):
                    collection.add(documents=documents, ids=ids)

                st.session_state["indexed"] = True
                st.session_state["doc_count"] = len(documents)
                st.sidebar.success("Indexed successfully!")

        except Exception as e:
            st.error(f"Indexing failed: {e}")

    st.divider()

    st.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-num">{st.session_state['doc_count']}</div>
            <div class="stat-lbl">Documents</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# 7. MAIN UI
# ──────────────────────────────────────────────
st.title("🔍 Semantic Similarity Search")
st.divider()

col_query, col_k = st.columns([3, 1])

with col_query:
    query = st.text_input("Query", placeholder="Search something…")

# ──────────────────────────────────────────────
# 8. FIXED SLIDER (NO CRASH VERSION)
# ──────────────────────────────────────────────
with col_k:
    doc_count = st.session_state.get("doc_count", 0)

    # SAFE BOUNDS
    if doc_count <= 1:
        max_k = 2
        slider_disabled = True
    else:
        max_k = doc_count
        slider_disabled = False

    default_k = min(3, max_k)

    if slider_disabled:
        st.slider("Top-K", 1, 2, 1, disabled=True)
        top_k = 1
    else:
        top_k = st.slider("Top-K", 1, max_k, default_k)

# ──────────────────────────────────────────────
# 9. SEARCH
# ──────────────────────────────────────────────
if query and st.session_state["doc_count"] > 0:
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=hf_ef,
        )

        with st.spinner("Searching…"):
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
            )

        docs = results["documents"][0]
        distances = results["distances"][0]

        if not docs:
            st.info("No results found.")
        else:
            st.markdown(f"### Results for: *{query}*")

            for i, (doc, dist) in enumerate(zip(docs, distances), 1):
                score = max(0, (1 - dist / 2) * 100)

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-doc"><b>#{i}</b> {doc}</div>
                        <div class="result-meta">
                            Distance: {dist:.4f}
                            <span class="badge">~{score:.1f}% match</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Search error: {e}")

else:
    st.info("Index documents first, then search.")