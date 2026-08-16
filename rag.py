# # from uuid import uuid4
# # from dotenv import load_dotenv
# # from pathlib import Path
# #
# # from langchain_community.document_loaders import UnstructuredURLLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_chroma import Chroma
# # from langchain_groq import ChatGroq
# # from langchain_huggingface import HuggingFaceEmbeddings
# #
# # from langchain.chains import create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain_core.prompts import ChatPromptTemplate
# #
# # load_dotenv()
# #
# # # Constants
# # CHUNK_SIZE = 1000
# # EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# # VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
# # COLLECTION_NAME = "real_estate"
# #
# # llm = None
# # vector_store = None
# #
# #
# # def initialize_components():
# #     global llm, vector_store
# #
# #     if llm is None:
# #         llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)
# #
# #     if vector_store is None:
# #         ef = HuggingFaceEmbeddings(
# #             model_name=EMBEDDING_MODEL,
# #             model_kwargs={"trust_remote_code": True}
# #         )
# #
# #         vector_store = Chroma(
# #             collection_name=COLLECTION_NAME,
# #             embedding_function=ef,
# #             persist_directory=str(VECTORSTORE_DIR)
# #         )
# #
# #
# # def process_urls(urls):
# #     """
# #     This function scraps data from a url and stores it in a vector db
# #     :param urls: input urls
# #     :return:
# #     """
# #     yield "Initializing Components"
# #     initialize_components()
# #
# #     yield "Resetting vector store...✅"
# #     vector_store.reset_collection()
# #
# #     yield "Loading data...✅"
# #     loader = UnstructuredURLLoader(urls=urls)
# #     data = loader.load()
# #
# #     yield "Splitting text into chunks...✅"
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         separators=["\n\n", "\n", ".", " "],
# #         chunk_size=CHUNK_SIZE
# #     )
# #     docs = text_splitter.split_documents(data)
# #
# #     yield "Add chunks to vector database...✅"
# #     uuids = [str(uuid4()) for _ in range(len(docs))]
# #     vector_store.add_documents(docs, ids=uuids)
# #
# #     yield "Done adding docs to vector database...✅"
# #
# # def generate_answer(query):
# #     if not vector_store:
# #         raise RuntimeError("Vector database is not initialized ")
# #
# #     chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
# #     result = chain.invoke({"question": query}, return_only_outputs=True)
# #     sources = result.get("sources", "")
# #
# #     return result['answer'], sources
# #
# #
# # if __name__ == "__main__":
# #     urls = [
# #         "https://www.selenium.dev/documentation/test_practices/encouraged/page_object_models/",
# #     ]
# #
# #     for step in process_urls(urls):
# #         print(step)
# #
# #     answer, sources = generate_answer(
# #         "what is Page Component Objects? Explain with examples"
# #     )
# #
# #     print(f"Answer: {answer}")
# #     print(f"Sources: {sources}")
#
#
# from uuid import uuid4
# from dotenv import load_dotenv
# from pathlib import Path
#
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
#
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
#
# from voiceAssisstentRag import generate_voice_answer
#
# load_dotenv()
#
# # Constants
# CHUNK_SIZE = 1000
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
# COLLECTION_NAME = "real_estate"
#
# llm = None
# vector_store = None
#
#
# def initialize_components():
#     global llm, vector_store
#
#     if llm is None:
#         llm = ChatGroq(
#             model="llama-3.3-70b-versatile",
#             temperature=0.7,
#             max_tokens=500
#         )
#
#     if vector_store is None:
#         embeddings = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL
#         )
#
#         VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
#
#         vector_store = Chroma(
#             collection_name=COLLECTION_NAME,
#             embedding_function=embeddings,
#             persist_directory=str(VECTORSTORE_DIR)
#         )
#
#
# def process_urls(urls):
#     """
#     Scrape data from URLs and store it in Chroma vector database.
#     """
#
#     yield "Initializing components..."
#     initialize_components()
#
#     yield "Loading data..."
#     loader = UnstructuredURLLoader(urls=urls)
#     data = loader.load()
#
#     if not data:
#         yield "No data found from the provided URLs."
#         return
#
#     yield "Splitting text into chunks..."
#
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ".", " "],
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=200
#     )
#
#     docs = text_splitter.split_documents(data)
#
#     yield f"Created {len(docs)} chunks..."
#
#     yield "Adding chunks to vector database..."
#
#     uuids = [str(uuid4()) for _ in docs]
#
#     vector_store.add_documents(
#         documents=docs,
#         ids=uuids
#     )
#
#     yield "Done adding documents to vector database...✅"
#
#
# def ask_voice_assistant(query):
#
#     answer, sources = generate_voice_answer(
#         query.strip()
#     )
#
#
# def generate_answer(query):
#     """
#     Generate an answer using RAG.
#     """
#
#     global llm, vector_store
#
#     if llm is None or vector_store is None:
#         initialize_components()
#
#     prompt = ChatPromptTemplate.from_template(
#         """
#         You are a helpful real estate research assistant.
#
#         Answer the user's question using ONLY the information provided
#         in the context below.
#
#         If the answer cannot be found in the context, say:
#         "I could not find this information in the provided sources."
#
#         <context>
#         {context}
#         </context>
#
#         Question:
#         {input}
#
#         Provide a clear and concise answer.
#         """
#     )
#
#     document_chain = create_stuff_documents_chain(
#         llm,
#         prompt
#     )
#
#     retrieval_chain = create_retrieval_chain(
#         vector_store.as_retriever(
#             search_kwargs={"k": 4}
#         ),
#         document_chain
#     )
#
#     result = retrieval_chain.invoke(
#         {
#             "input": query
#         }
#     )
#
#     answer = result.get("answer", "")
#
#     # Extract sources from retrieved documents
#     sources = []
#
#     for doc in result.get("context", []):
#         source = doc.metadata.get("source")
#
#         if source and source not in sources:
#             sources.append(source)
#
#     sources_text = "\n".join(sources)
#
#     return answer, sources_text
#
#
# if __name__ == "__main__":
#
#     urls = [
#         "https://www.selenium.dev/documentation/test_practices/encouraged/page_object_models/"
#     ]
#
#     for step in process_urls(urls):
#         print(step)
#
#     answer, sources = generate_answer(
#         "What is Page Component Objects? Explain with examples."
#     )
#
#     print(f"\nAnswer:\n{answer}")
#     print(f"\nSources:\n{sources}")