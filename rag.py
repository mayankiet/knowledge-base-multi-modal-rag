from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Add chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources


if __name__ == "__main__":
    urls = [
        "https://www.selenium.dev/documentation/test_practices/encouraged/page_object_models/",
    ]

    for step in process_urls(urls):
        print(step)

    answer, sources = generate_answer(
        "what is Page Component Objects? Explain with examples"
    )

    print(f"Answer: {answer}")
    print(f"Sources: {sources}")