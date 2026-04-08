"""
Streamlit UI for Logistics Document AI.
  Terminal 1: cd backend && python main.py
  Terminal 2: streamlit run frontend/streamlit_app.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import requests
import streamlit as st

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logistics Document AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "page":                    "dashboard",   # dashboard | rag_train | rag_ask | extract
    "collections":             [],
    "current_doc_id":          None,
    "current_filename":        None,
    "answer_data":             None,
    "direct_extract_result":   None,
    "direct_extract_filename": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Navigation helpers ─────────────────────────────────────────────────────────
def go(page: str):
    st.session_state.page = page
    # clear stale results when navigating away
    st.session_state.answer_data = None
    st.session_state.direct_extract_result = None
    st.session_state.direct_extract_filename = None

# ── API helper ─────────────────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs):
    try:
        r = getattr(requests, method)(f"{BACKEND_URL}{path}", timeout=60, **kwargs)
        if r.ok:
            return r.json(), None
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        return None, str(detail)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach backend at {BACKEND_URL}. Is `python main.py` running?"
    except Exception as e:
        return None, str(e)

# ── Shared helpers ─────────────────────────────────────────────────────────────
def load_collections():
    data, err = api("get", "/docs")
    st.session_state.collections = data or [] if not err else []
    return err


def confidence_color(score: float) -> str:
    return "🟢" if score >= 0.75 else "🟡" if score >= 0.5 else "🔴"


def chunk_type_badge(ct: str) -> str:
    return {"kv_block": "🟩", "table_row": "🟨", "narrative": "🟦"}.get(ct, "⬜")


def render_extraction_table(d: dict):
    rows = [
        ("Shipment ID",        d.get("shipment_id")),
        ("Shipper Name",       (d.get("shipper") or {}).get("name")),
        ("Shipper Address",    (d.get("shipper") or {}).get("address")),
        ("Consignee Name",     (d.get("consignee") or {}).get("name")),
        ("Consignee Address",  (d.get("consignee") or {}).get("address")),
        ("Pickup Date/Time",   d.get("pickup_datetime")),
        ("Delivery Date/Time", d.get("delivery_datetime")),
        ("Equipment Type",     d.get("equipment_type")),
        ("Mode",               d.get("mode")),
        ("Rate",               d.get("rate")),
        ("Currency",           d.get("currency")),
        ("Weight",             d.get("weight")),
        ("Carrier Name",       d.get("carrier_name")),
    ]
    found   = [(k, v) for k, v in rows if v is not None]
    missing = [(k, v) for k, v in rows if v is None]
    if found:
        for field, val in found:
            c1, c2 = st.columns([2, 3])
            c1.markdown(f"**{field}**")
            c2.markdown(str(val))
            st.divider()
    if missing:
        with st.expander(f"⬜ {len(missing)} field(s) not found"):
            for field, _ in missing:
                st.caption(f"— {field}")


def back_button(label: str = "← Back to Dashboard"):
    if st.button(label, key="back_btn"):
        go("dashboard")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "dashboard":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## ✈️ Logistics Document AI")
    st.caption("Choose what you want to do")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📚 RAG Chatbot")
        st.write("Train on logistics documents and ask natural language questions grounded in document content.")
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("🗂️ Train", use_container_width=True, type="primary"):
            go("rag_train")
            st.rerun()
        if c2.button("💬 Ask Queries", use_container_width=True):
            load_collections()
            go("rag_ask")
            st.rerun()

    with col2:
        st.markdown("### 📤 Extract")
        st.write("Upload any logistics document and instantly extract structured shipment fields as JSON — nothing is stored.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📤 Go to Extract", use_container_width=True):
            go("extract")
            st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RAG — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "rag_train":
    back_button()
    st.markdown("## 🗂️ Train — Upload Documents")
    st.caption("Upload PDFs, DOCX, or TXT files to index them into the knowledge base. Documents persist across sessions.")
    st.divider()

    # Load existing collections
    if not st.session_state.collections:
        load_collections()

    uploaded_files = st.file_uploader(
        "Select one or more files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="train_uploader",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
        if st.button("Upload & Index All", type="primary"):
            progress = st.progress(0, text="Starting…")
            for i, f in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"Uploading {f.name}…")
                data, err = api(
                    "post", "/upload",
                    files={"file": (f.name, f.getvalue(), f.type)},
                )
                if err:
                    st.error(f"✗ {f.name}: {err}")
                else:
                    st.success(f"✓ **{data['filename']}** — {data['chunk_count']} chunks, {data['page_count']} page(s)")
            progress.progress(1.0, text="Done")
            load_collections()
            st.rerun()

    # Show already-indexed documents
    st.divider()
    col_count = len(st.session_state.collections)
    st.markdown(f"#### Indexed Documents ({col_count})")

    if col_count == 0:
        st.caption("No documents indexed yet.")
    else:
        for doc in st.session_state.collections:
            c1, c2, c3 = st.columns([5, 2, 1])
            ts = doc.get("upload_timestamp", "")
            c1.markdown(f"**{doc['filename']}**")
            c2.caption(f"{doc['chunk_count']} chunks · {doc['page_count']}p" + (f" · {ts[:10]}" if ts else ""))
            if c3.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete"):
                _, err = api("delete", f"/docs/{doc['doc_id']}")
                if err:
                    st.error(err)
                else:
                    if st.session_state.current_doc_id == doc["doc_id"]:
                        st.session_state.current_doc_id = None
                        st.session_state.current_filename = None
                    load_collections()
                    st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RAG — ASK QUERIES
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "rag_ask":
    back_button()
    st.markdown("## 💬 Ask Queries")
    st.divider()

    if not st.session_state.collections:
        load_collections()

    col_count = len(st.session_state.collections)

    if col_count == 0:
        st.warning("No documents trained yet. Go to **Train** first to upload documents.", icon="⚠️")
        if st.button("Go to Train →", type="primary"):
            go("rag_train")
            st.rerun()
        st.stop()

    # Document selector
    doc_options = {doc["filename"]: doc["doc_id"] for doc in st.session_state.collections}
    selected_filename = st.selectbox(
        "Select a document to query",
        options=list(doc_options.keys()),
        index=0 if st.session_state.current_filename not in doc_options else list(doc_options.keys()).index(st.session_state.current_filename),
    )
    doc_id = doc_options[selected_filename]
    st.session_state.current_doc_id = doc_id
    st.session_state.current_filename = selected_filename

    st.markdown("<br>", unsafe_allow_html=True)

    # Question input
    with st.form("ask_form"):
        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the carrier rate?  Who is the consignee?  When is pickup?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("Searching document…"):
            data, err = api("post", "/ask", json={"doc_id": doc_id, "question": question})
        if err:
            st.error(err)
        else:
            st.session_state.answer_data = data

    # Answer display
    if st.session_state.answer_data:
        d    = st.session_state.answer_data
        conf = d["confidence"]
        icon = confidence_color(conf)
        lbl  = "High" if conf >= 0.75 else "Medium" if conf >= 0.5 else "Low"

        st.divider()
        c1, c2 = st.columns([4, 1])
        c1.markdown("**Answer**")
        c2.markdown(f"{icon} **{conf*100:.1f}%** {lbl}")

        if d.get("guardrail_triggered"):
            st.warning(d["answer"], icon="⚠️")
        else:
            st.success(d["answer"])

        st.progress(conf)

        if d.get("source_chunks"):
            with st.expander(f"📎 {len(d['source_chunks'])} source chunk(s)"):
                for i, chunk in enumerate(d["source_chunks"], 1):
                    st.markdown(
                        f"**#{i}** &nbsp; {chunk_type_badge(chunk['chunk_type'])} `{chunk['chunk_type']}` &nbsp;"
                        f"Page {chunk['page_number']} &nbsp; `{chunk['similarity']*100:.1f}%`"
                    )
                    st.code(chunk["text"], language=None)
                    if i < len(d["source_chunks"]):
                        st.divider()

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIRECT EXTRACT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "extract":
    back_button()
    st.markdown("## 📤 Extract Structured Fields")
    st.caption("Upload a file to extract shipment fields instantly. Nothing is stored or indexed.")
    st.divider()

    direct_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        key="extract_uploader",
    )

    c1, c2 = st.columns([2, 1])
    run = c1.button("Extract Fields", type="primary", disabled=direct_file is None)
    if c2.button("Clear", disabled=st.session_state.direct_extract_result is None):
        st.session_state.direct_extract_result = None
        st.session_state.direct_extract_filename = None
        st.rerun()

    if run and direct_file:
        with st.spinner(f"Extracting from {direct_file.name}…"):
            data, err = api(
                "post", "/extract-file",
                files={"file": (direct_file.name, direct_file.getvalue(), direct_file.type)},
            )
        if err:
            st.error(err)
        else:
            st.session_state.direct_extract_result = data.get("data")
            st.session_state.direct_extract_filename = direct_file.name

    if st.session_state.direct_extract_result:
        st.success(f"✓ {st.session_state.direct_extract_filename}")
        view = st.radio("View as", ["Table", "JSON"], horizontal=True, key="extract_view")
        st.divider()
        if view == "JSON":
            st.json(st.session_state.direct_extract_result)
        else:
            render_extraction_table(st.session_state.direct_extract_result)

    st.stop()
