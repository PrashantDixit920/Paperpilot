import streamlit as st

from services.document_loader import load_document
from services.vector_store import create_retriever
from services.rag_chain import create_qa_chain, create_summary_chain


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PaperPilot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MINIMAL DARK CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #0E1117;
        }

        .block-container {
            max-width: 1050px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        h1, h2, h3 {
            letter-spacing: -0.4px;
        }

        .brand {
            font-size: 1.8rem;
            font-weight: 650;
            letter-spacing: -0.8px;
            margin-bottom: 0;
        }

        .subtitle {
            color: #9299A8;
            font-size: 0.95rem;
            margin-top: 0.25rem;
            margin-bottom: 1.5rem;
        }

        .muted {
            color: #9299A8;
            font-size: 0.85rem;
        }

        div[data-testid="stFileUploader"] {
            border: 1px solid #292E39;
            border-radius: 10px;
            padding: 0.5rem;
            background: #141821;
        }

        div[data-testid="stTabs"] button {
            font-size: 0.95rem;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid #252A35;
            border-radius: 10px;
            margin-bottom: 0.6rem;
        }

        div[data-testid="stMetric"] {
            background: #141821;
            border: 1px solid #292E39;
            border-radius: 10px;
            padding: 0.8rem;
        }

        div.stButton > button {
            border-radius: 8px;
            min-height: 2.5rem;
        }

        hr {
            border-color: #292E39;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "retriever": None,
    "qa_chain": None,
    "summary": None,
    "documents": None,
    "processed_files": [],
    "messages": [],
    "document_signature": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_document_state():
    st.session_state.retriever = None
    st.session_state.qa_chain = None
    st.session_state.summary = None
    st.session_state.documents = None
    st.session_state.processed_files = []
    st.session_state.messages = []
    st.session_state.document_signature = None


def extract_answer_text(response):
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

        return "\n".join(parts).strip()

    return str(content).strip()


def get_source_details(documents):
    sources = []

    for document in documents:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        page_label = (
            f"Page {page + 1}"
            if isinstance(page, int)
            else "Page unavailable"
        )

        entry = {
            "source": source,
            "page": page_label,
        }

        if entry not in sources:
            sources.append(entry)

    return sources


def render_summary(summary):
    if summary.title:
        st.subheader(summary.title)

    st.markdown("#### Overview")
    st.write(summary.overview)

    st.markdown("#### Key findings")
    for finding in summary.key_findings:
        st.markdown(f"- {finding}")

    st.markdown("#### Methodology")
    st.write(summary.methodology)

    st.markdown("#### Limitations")
    for limitation in summary.limitations:
        st.markdown(f"- {limitation}")

    st.markdown("#### Conclusion")
    st.write(summary.conclusion)

    st.download_button(
        "Download summary",
        data=summary.model_dump_json(indent=2),
        file_name="paperpilot_summary.json",
        mime="application/json",
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📄 PaperPilot")
    st.caption("Document research assistant")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("Reset all", use_container_width=True):
            reset_document_state()
            st.rerun()

    if st.session_state.processed_files:
        st.divider()
        st.markdown("**Processed files**")

        for filename in st.session_state.processed_files:
            st.caption(f"📄 {filename}")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="brand">PaperPilot</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">'
    'Summarize documents. Ask questions. Find answers.'
    '</p>',
    unsafe_allow_html=True,
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:
    current_signature = tuple(
        (file.name, file.size, file.getvalue())
        for file in uploaded_files
    )

    if (
        st.session_state.document_signature is not None
        and current_signature != st.session_state.document_signature
    ):
        reset_document_state()

    if process_button:
        reset_document_state()

        try:
            documents = []

            with st.spinner("Extracting document text..."):
                for uploaded_file in uploaded_files:
                    documents.extend(load_document(uploaded_file))

            if not documents:
                st.error("No readable text was found.")
                st.stop()

            with st.spinner("Building the document index..."):
                retriever = create_retriever(documents)
                qa_chain = create_qa_chain(retriever)

            st.session_state.documents = documents
            st.session_state.retriever = retriever
            st.session_state.qa_chain = qa_chain
            st.session_state.processed_files = [
                file.name for file in uploaded_files
            ]
            st.session_state.document_signature = current_signature

            st.success("Documents are ready.")

        except Exception as error:
            reset_document_state()
            st.error("Could not process the documents.")

            with st.expander("Error details"):
                st.code(str(error))


# ============================================================
# DOCUMENT STATUS
# ============================================================

if st.session_state.documents:
    st.caption(
        f"{len(st.session_state.processed_files)} document(s) · "
        f"{len(st.session_state.documents)} pages/sections processed"
    )


# ============================================================
# MAIN TABS
# ============================================================

summary_tab, chat_tab, documents_tab = st.tabs(
    ["Summary", "Chat", "Documents"]
)


# ============================================================
# SUMMARY TAB
# ============================================================

with summary_tab:
    st.markdown("### Document summary")

    if not st.session_state.documents:
        st.caption("Upload and process documents to get started.")

    else:
        if st.button("Generate summary", type="primary"):
            try:
                with st.spinner("Summarizing documents..."):
                    summary_chain = create_summary_chain(
                        st.session_state.documents
                    )
                    summary = summary_chain.invoke({})

                st.session_state.summary = summary
                st.success("Summary ready.")

            except Exception as error:
                st.error("Summary generation failed.")

                with st.expander("Error details"):
                    st.code(str(error))

        if st.session_state.summary is not None:
            st.divider()
            render_summary(st.session_state.summary)


# ============================================================
# CHAT TAB
# ============================================================

# ============================================================
# CHAT TAB
# ============================================================

with chat_tab:
    st.markdown("### Ask your documents")

    if not st.session_state.qa_chain:
        st.caption("Process documents before starting a conversation.")

    else:
        # Display existing conversation.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.caption(
                                f'{source["source"]} · {source["page"]}'
                            )

        question = st.chat_input(
            "Ask a question...",
            key="paperpilot_chat_input",
        )

        if question:
            # Capture previous messages before adding the current question.
            chat_history = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in st.session_state.messages
                if message.get("role") in ("user", "assistant")
            ]

            # Store the user's question.
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching documents..."):
                        # Pass a dictionary, not a plain string.
                        result = st.session_state.qa_chain.invoke(
                            {
                                "question": question,
                                "chat_history": chat_history,
                            }
                        )

                        response = result["answer"]
                        retrieved_documents = result["source_documents"]

                        answer = extract_answer_text(response)

                    if not answer:
                        answer = (
                            "No answer was generated. Try another question."
                        )

                    sources = get_source_details(retrieved_documents)

                    st.markdown(answer)

                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.caption(
                                    f'{source["source"]} · {source["page"]}'
                                )

                    # Save the answer for future follow-up questions.
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:
                    st.error("Unable to answer your question.")

                    with st.expander("Error details"):
                        st.code(str(error))
# ============================================================
# DOCUMENTS TAB
# ============================================================

with documents_tab:
    st.markdown("### Document contents")

    if not st.session_state.documents:
        st.caption("Processed document information will appear here.")

    else:
        selected_index = st.selectbox(
            "Select a section",
            options=range(len(st.session_state.documents)),
            format_func=lambda index: (
                f"Section {index + 1} · "
                f"{st.session_state.documents[index].metadata.get('source', 'Unknown')}"
            ),
        )

        document = st.session_state.documents[selected_index]

        st.caption(
            f"Source: {document.metadata.get('source', 'Unknown')}"
        )

        page = document.metadata.get("page")
        if isinstance(page, int):
            st.caption(f"Page {page + 1}")

        st.text_area(
            "Extracted text",
            value=document.page_content,
            height=350,
            disabled=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<p class="muted" style="text-align:center;">'
    'PaperPilot · AI document research'
    '</p>',
    unsafe_allow_html=True,
)