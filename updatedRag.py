from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from atlassian import Confluence


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
vector_store = None


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

def initialize_components():

    global llm, vector_store

    if llm is None:

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=500
        )

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

def load_confluence_page(
    page_id
):

    print(
        f"Loading Confluence page: {page_id}"
    )

    confluence = Confluence(

        url=os.getenv(
            "CONFLUENCE_URL"
        ),

        username=os.getenv(
            "CONFLUENCE_USERNAME"
        ),

        password=os.getenv(
            "CONFLUENCE_API_TOKEN"
        ),

        cloud=True
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

    # Convert HTML into text
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        html_content,
        "html.parser"
    )

    text = soup.get_text(
        separator="\n"
    )

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

def load_confluence_pages(
    page_ids
):

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
        f"Loaded {len(web_docs)} web documents...✅"
    )


    # --------------------------------------------------------
    # LOAD PDF DATA
    # --------------------------------------------------------

    yield "Loading PDF documents..."

    pdf_docs = load_pdfs(
        pdf_paths or []
    )

    yield (
        f"Loaded {len(pdf_docs)} PDF pages...✅"
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
# RAG ANSWER
# ============================================================

def generate_answer(query):

    global llm, vector_store

    if (
        llm is None
        or vector_store is None
    ):

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


    result = retrieval_chain.invoke(

        {
            "input": query
        }
    )


    answer = result.get(
        "answer",
        ""
    )


    # --------------------------------------------------------
    # EXTRACT SOURCES
    # --------------------------------------------------------

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
                f"[Confluence] {title} - {source}"
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


    sources_text = "\n".join(
        sources
    )


    return answer, sources_text


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    urls = [

        "https://www.selenium.dev/documentation/webdriver/"
    ]


    pdf_paths = [

        "resources/document/selenium.pdf"
    ]


    confluence_page_ids = [

        "mayanksdet"
    ]


    for step in process_documents(

        urls=urls,

        pdf_paths=pdf_paths,

        confluence_page_ids=confluence_page_ids

    ):

        print(step)


    answer, sources = generate_answer(

        "What is webdriver?"
    )


    print(
        f"\nAnswer:\n{answer}"
    )

    print(
        f"\nSources:\n{sources}"
    )