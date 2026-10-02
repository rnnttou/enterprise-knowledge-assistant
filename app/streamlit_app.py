"""Step 6: Streamlit UI for the question-answering API.

Run with:
    streamlit run app/streamlit_app.py

This is a thin HTTP client over the FastAPI backend (app/api.py) - it never
touches Gemini or ChromaDB directly, so the backend must be running too:
    uvicorn app.api:app --reload
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚")
st.title("📚 Enterprise Knowledge Assistant")
st.caption("Ask questions about your company's documents. Answers are grounded in cited excerpts.")

with st.sidebar:
    st.header("Settings")
    groups_input = st.text_input(
        "Groups (comma-separated)",
        value="",
        help="Leave empty for public documents only, e.g. 'hr, legal'.",
    )
    k = st.slider("Chunks to retrieve", min_value=1, max_value=20, value=5)
    st.divider()
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
    if st.button("Re-index documents"):
        with st.spinner("Indexing data/ ..."):
            try:
                response = httpx.post(f"{API_URL}/index", timeout=120)
                response.raise_for_status()
                stats = response.json()
                st.success(
                    f"Indexed {len(stats['indexed'])} file(s), added {stats['chunks_added']} chunk(s)."
                )
            except httpx.HTTPError as exc:
                st.error(f"Indexing failed: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    groups = [g.strip() for g in groups_input.split(",") if g.strip()]

    with st.chat_message("assistant"):
        try:
            response = httpx.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "groups": groups,
                    "k": k,
                    "conversation_id": st.session_state.conversation_id,
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Couldn't reach the API: {exc}")
        else:
            st.session_state.conversation_id = result["conversation_id"]
            st.markdown(result["answer"])
            if result["citations"]:
                with st.expander("Sources"):
                    for citation in result["citations"]:
                        st.markdown(
                            f"**[{citation['marker']}] {citation['source']}, p. {citation['page']}**  \n"
                            f"{citation['snippet']}..."
                        )
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
