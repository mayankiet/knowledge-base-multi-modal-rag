import streamlit as st
import tempfile
from gtts import gTTS

from voiceAssisstentRag import (
    process_documents,
    generate_answer,
    ask_voice_assistant
)


st.set_page_config(
    page_title="Knowledge Base Voice Assistant",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Knowledge Base Voice Assistant")
st.caption("Search across Web URLs, PDF documents and Confluence")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📚 Knowledge Sources")


# ============================================================
# WEB URLS
# ============================================================

st.sidebar.subheader("🌐 Web URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")


# ============================================================
# PDF UPLOAD
# ============================================================

st.sidebar.subheader("📄 PDF Documents")

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# CONFLUENCE
# ============================================================

st.sidebar.subheader("🔷 Confluence")

confluence_pages = st.sidebar.text_area(
    "Confluence Page IDs",
    placeholder="123456789\n987654321",
    help="Enter one Confluence page ID per line"
)


# ============================================================
# PROCESS BUTTON
# ============================================================

process_button = st.sidebar.button(
    "🚀 Process Knowledge Base",
    use_container_width=True
)


status_placeholder = st.empty()


# ============================================================
# PROCESS KNOWLEDGE BASE
# ============================================================

if process_button:

    urls = [
        url
        for url in [url1, url2, url3]
        if url.strip()
    ]


    # --------------------------------------------------------
    # SAVE UPLOADED PDFs
    # --------------------------------------------------------

    pdf_paths = []

    if uploaded_pdfs:

        pdf_directory = "resources/documents"

        import os

        os.makedirs(
            pdf_directory,
            exist_ok=True
        )

        for uploaded_file in uploaded_pdfs:

            pdf_path = os.path.join(
                pdf_directory,
                uploaded_file.name
            )

            with open(
                pdf_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            pdf_paths.append(
                pdf_path
            )


    # --------------------------------------------------------
    # CONFLUENCE PAGE IDs
    # --------------------------------------------------------

    confluence_page_ids = [
        page_id.strip()
        for page_id in confluence_pages.splitlines()
        if page_id.strip()
    ]


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not urls and not pdf_paths and not confluence_page_ids:

        status_placeholder.error(
            "Please provide at least one URL, PDF or Confluence page ID."
        )

    else:

        try:

            for status in process_documents(

                urls=urls,

                pdf_paths=pdf_paths,

                confluence_page_ids=confluence_page_ids

            ):

                status_placeholder.info(
                    status
                )

            status_placeholder.success(
                "✅ Knowledge base processed successfully!"
            )

        except Exception as e:

            status_placeholder.error(
                f"Error while processing documents: {e}"
            )


# ============================================================
# QUESTION
# ============================================================

st.divider()

st.subheader("💬 Ask Your Question")

query = st.text_input(
    "Enter your question:",
    placeholder="What is webdriver?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask",
    use_container_width=True
) and query.strip():

    try:

        # ----------------------------------------------------
        # TEXT RAG
        # ----------------------------------------------------

        with st.spinner("Searching knowledge base..."):

            answer, sources = generate_answer(
                query.strip()
            )


        # ----------------------------------------------------
        # TEXT ANSWER
        # ----------------------------------------------------

        st.subheader("🤖 Answer")

        st.write(
            answer
        )


        # ----------------------------------------------------
        # VOICE ANSWER
        # ----------------------------------------------------

        with st.spinner("Generating voice response..."):

            voice_answer, voice_sources = (
                ask_voice_assistant(
                    query.strip()
                )
            )


        st.subheader("🔊 Voice Answer")

        st.write(
            voice_answer
        )


        # ----------------------------------------------------
        # TEXT TO SPEECH
        # ----------------------------------------------------

        tts = gTTS(
            text=voice_answer,
            lang="en",
            slow=False
        )


        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )


        tts.save(
            audio_file.name
        )


        # ----------------------------------------------------
        # AUDIO PLAYER
        # ----------------------------------------------------

        st.audio(
            audio_file.name,
            format="audio/mp3"
        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if sources:

            st.subheader("📚 Sources")

            for source in sources.split("\n"):

                st.write(
                    f"• {source}"
                )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )