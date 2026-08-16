from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os
import re

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from atlassian import Confluence


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

COLLECTION_NAME = "automation_research"


# ============================================================
# GLOBAL COMPONENTS
# ============================================================

llm = None
voice_llm = None
vector_store = None


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

def initialize_components():

    global llm, voice_llm, vector_store

    # --------------------------------------------------------
    # TEXT LLM
    # --------------------------------------------------------

    if llm is None:

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )

    # --------------------------------------------------------
    # VOICE LLM
    # --------------------------------------------------------

    if voice_llm is None:

        voice_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=250
        )

    # --------------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------------

    if vector_store is None:

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        VECTORSTORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )


# ============================================================
# LOAD WEB URLS
# ============================================================

def load_urls(urls):

    if not urls:
        return []

    print("Loading web URLs...")

    loader = UnstructuredURLLoader(
        urls=urls
    )

    documents = loader.load()

    for doc in documents:
        doc.metadata["source_type"] = "web"

    return documents


# ============================================================
# LOAD PDF FILES
# ============================================================

def load_pdfs(pdf_paths):

    documents = []

    if not pdf_paths:
        return documents

    print("Loading PDF files...")

    for pdf_path in pdf_paths:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():

            print(
                f"PDF not found: {pdf_path}"
            )

            continue

        loader = PyPDFLoader(
            str(pdf_path)
        )

        pdf_docs = loader.load()

        for doc in pdf_docs:

            doc.metadata["source_type"] = "pdf"

            doc.metadata["source"] = pdf_path.name

            doc.metadata["file_name"] = pdf_path.name

        documents.extend(pdf_docs)

    return documents


# ============================================================
# LOAD CONFLUENCE PAGE
# ============================================================

def load_confluence_page(page_id):

    print(
        f"Loading Confluence page: {page_id}"
    )

    # --------------------------------------------------------
    # SELF-HOSTED CONFLUENCE
    # --------------------------------------------------------

    confluence = Confluence(

        url=os.getenv(
            "CONFLUENCE_URL"
        ),

        token=os.getenv(
            "CONFLUENCE_API_TOKEN"
        ),

        cloud=False
    )

    page = confluence.get_page_by_id(
        page_id,
        expand="body.storage,version,space"
    )

    if not page:

        raise RuntimeError(
            f"Confluence page not found: {page_id}"
        )

    title = page["title"]

    html_content = (
        page["body"]["storage"]["value"]
    )

    # --------------------------------------------------------
    # HTML -> TEXT
    # --------------------------------------------------------

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        html_content,
        "html.parser"
    )

    text = soup.get_text(
        separator="\n"
    )

    # --------------------------------------------------------
    # CREATE DOCUMENT
    # --------------------------------------------------------

    from langchain_core.documents import Document

    document = Document(

        page_content=text,

        metadata={

            "source_type": "confluence",

            "source": (
                f"{os.getenv('CONFLUENCE_URL')}"
                f"/pages/{page_id}"
            ),

            "page_id": page_id,

            "title": title,

            "space": page.get(
                "space",
                {}
            ).get(
                "name",
                ""
            )
        }
    )

    return [document]


# ============================================================
# LOAD MULTIPLE CONFLUENCE PAGES
# ============================================================

def load_confluence_pages(page_ids):

    documents = []

    if not page_ids:
        return documents

    for page_id in page_ids:

        try:

            docs = load_confluence_page(
                page_id
            )

            documents.extend(docs)

        except Exception as e:

            print(
                f"Error loading Confluence page "
                f"{page_id}: {e}"
            )

    return documents


# ============================================================
# PROCESS ALL SOURCES
# ============================================================

def process_documents(
    urls=None,
    pdf_paths=None,
    confluence_page_ids=None
):

    yield "Initializing components...✅"

    initialize_components()

    # --------------------------------------------------------
    # LOAD WEB DATA
    # --------------------------------------------------------

    yield "Loading web URLs..."

    web_docs = load_urls(
        urls or []
    )

    yield (
        f"Loaded {len(web_docs)} "
        f"web documents...✅"
    )

    # --------------------------------------------------------
    # LOAD PDF DATA
    # --------------------------------------------------------

    yield "Loading PDF documents..."

    pdf_docs = load_pdfs(
        pdf_paths or []
    )

    yield (
        f"Loaded {len(pdf_docs)} "
        f"PDF pages...✅"
    )

    # --------------------------------------------------------
    # LOAD CONFLUENCE DATA
    # --------------------------------------------------------

    yield "Loading Confluence pages..."

    confluence_docs = load_confluence_pages(
        confluence_page_ids or []
    )

    yield (
        f"Loaded {len(confluence_docs)} "
        f"Confluence pages...✅"
    )

    # --------------------------------------------------------
    # COMBINE DOCUMENTS
    # --------------------------------------------------------

    all_documents = (
        web_docs
        + pdf_docs
        + confluence_docs
    )

    if not all_documents:

        yield "No documents found."

        return

    yield (
        f"Total documents loaded: "
        f"{len(all_documents)}"
    )

    # --------------------------------------------------------
    # SPLIT INTO CHUNKS
    # --------------------------------------------------------

    yield "Splitting documents into chunks..."

    text_splitter = RecursiveCharacterTextSplitter(

        separators=[
            "\n\n",
            "\n",
            ".",
            " ",
            ""
        ],

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP
    )

    docs = text_splitter.split_documents(
        all_documents
    )

    yield (
        f"Created {len(docs)} chunks...✅"
    )

    # --------------------------------------------------------
    # ADD TO CHROMA
    # --------------------------------------------------------

    yield "Adding documents to Chroma..."

    uuids = [
        str(uuid4())
        for _ in docs
    ]

    vector_store.add_documents(

        documents=docs,

        ids=uuids
    )

    yield (
        "Done adding documents "
        "to vector database...✅"
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def process_urls(urls):

    return process_documents(
        urls=urls
    )


# ============================================================
# CREATE RETRIEVAL CHAIN
# ============================================================

def get_retrieval_chain():

    initialize_components()

    prompt = ChatPromptTemplate.from_template(

        """
        You are an AI research assistant.

        Answer the user's question using ONLY
        the information available in the context.

        The context can come from:

        - Web pages
        - PDF documents
        - Confluence pages

        If the answer cannot be found in the
        provided context, say:

        "I could not find this information
        in the provided sources."

        Always provide a concise and accurate answer.

        Context:

        {context}

        Question:

        {input}
        """
    )

    document_chain = create_stuff_documents_chain(

        llm,

        prompt
    )

    retriever = vector_store.as_retriever(

        search_kwargs={
            "k": 5
        }
    )

    retrieval_chain = create_retrieval_chain(

        retriever,

        document_chain
    )

    return retrieval_chain


# ============================================================
# NORMAL TEXT RAG ANSWER
# ============================================================

def generate_answer(query):

    retrieval_chain = get_retrieval_chain()

    result = retrieval_chain.invoke(
        {
            "input": query
        }
    )

    answer = result.get(
        "answer",
        ""
    )

    sources = extract_sources(
        result
    )

    return answer, sources


# ============================================================
# EXTRACT SOURCES
# ============================================================

def extract_sources(result):

    sources = []

    for doc in result.get(
        "context",
        []
    ):

        source_type = doc.metadata.get(
            "source_type",
            "unknown"
        )

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        title = doc.metadata.get(
            "title",
            ""
        )

        if source_type == "confluence":

            source_info = (
                f"[Confluence] "
                f"{title} - {source}"
            )

        elif source_type == "pdf":

            source_info = (
                f"[PDF] {source}"
            )

        elif source_type == "web":

            source_info = (
                f"[Web] {source}"
            )

        else:

            source_info = source

        if source_info not in sources:

            sources.append(
                source_info
            )

    return "\n".join(sources)


# ============================================================
# VOICE ASSISTANT RAG
# ============================================================

def generate_voice_answer(query):

    initialize_components()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    documents = retriever.invoke(query)

    if not documents:
        return (
            "Sorry, I could not find that information.",
            ""
        )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    voice_prompt = f"""
You are a helpful AI voice assistant.

Answer the user's question using ONLY the information
provided below.

Rules:
- Keep the answer short.
- Use 2 to 5 sentences.
- Make it conversational.
- Do not use Markdown.
- Do not use bullet points.
- Do not mention RAG.
- Do not mention vector databases.
- Do not mention context.
- If the answer is not available, say:
  "Sorry, I could not find that information."

Information:

{context}

User question:

{query}

Answer:
"""

    try:

        response = voice_llm.invoke(
            voice_prompt
        )

        answer = response.content

        answer = clean_voice_response(
            answer
        )

    except Exception as e:

        print(
            f"Voice LLM error: {e}"
        )

        return (
            "Sorry, I was unable to generate a voice answer.",
            ""
        )

    sources = extract_sources({
        "context": documents
    })

    return answer, sources


# ============================================================
# CLEAN VOICE RESPONSE
# ============================================================

def clean_voice_response(text):

    if not text:
        return ""

    # Remove markdown headings
    text = re.sub(
        r"#+\s*",
        "",
        text
    )

    # Remove bullet points
    text = re.sub(
        r"^\s*[-*•]\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove markdown bold
    text = text.replace(
        "**",
        ""
    )

    # Remove markdown italic
    text = text.replace(
        "*",
        ""
    )

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# VOICE ASSISTANT HELPER
# ============================================================


def ask_voice_assistant(query):

    if not query or not query.strip():

        return (
            "Sorry, I didn't hear a question.",
            ""
        )

    return generate_voice_answer(
        query.strip()
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    urls = [

        "https://www.selenium.dev/"

    ]

    pdf_paths = [

        "resources/document/selenium.pdf"

    ]

    confluence_page_ids = [

        "123456789"

    ]

    # --------------------------------------------------------
    # INGEST DATA
    # --------------------------------------------------------

    for step in process_documents(

        urls=urls,

        pdf_paths=pdf_paths,

        confluence_page_ids=confluence_page_ids

    ):

        print(step)

    # --------------------------------------------------------
    # NORMAL TEXT QUESTION
    # --------------------------------------------------------

    answer, sources = generate_answer(
        "What is Page Component Objects?"
    )

    print(
        f"\nAnswer:\n{answer}"
    )

    print(
        f"\nSources:\n{sources}"
    )

    # --------------------------------------------------------
    # VOICE QUESTION
    # --------------------------------------------------------

    voice_answer, voice_sources = ask_voice_assistant(
        "What is WebDriver?"
    )

    print(
        f"\nVoice Answer:\n{voice_answer}"
    )

    print(
        f"\nVoice Sources:\n{voice_sources}"
    )