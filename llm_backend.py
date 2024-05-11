from os import environ, path
from typing import List
import chromadb
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Constants
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "Llama3-70b-8192"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 8192
DOCUMENT_DIR = "./documents/"
FILE_NAME = "Stability in Smart Grids.pdf"
VECTOR_STORE_DIR = "./vectorstore/"
COLLECTION_NAME = "collection1"

def load_document() -> List[Document]:
    """Loads the PDF file specified by the FILE_NAME constant."""
    pdf = PyPDFLoader(path.join(DOCUMENT_DIR, FILE_NAME)).load()
    return pdf

def chunk_document(documents: List[Document]) -> List[Document]:
    """Splits the input documents into maximum of CHUNK_SIZE chunks."""
    tokenizer = AutoTokenizer.from_pretrained("jinaai/" + EMBED_MODEL_NAME, cache_dir=environ.get("HF_HOME"))
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_SIZE // 50)
    return text_splitter.split_documents(documents)

def get_vectorstore_retriever(embedding_model: JinaEmbeddings) -> VectorStoreRetriever:
    """Returns the vectorstore."""
    db = chromadb.PersistentClient(VECTOR_STORE_DIR)
    try:
        # Check for the existence of the vectorstore specified by the COLLECTION_NAME
        db.get_collection(COLLECTION_NAME)
        retriever = Chroma(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        ).as_retriever(search_kwargs={"k": 3})
    except:
        # The vectorstore doesn't exist, so create it.
        pdf = load_document()
        chunks = chunk_document(pdf)
        retriever = create_and_store_embeddings(embedding_model, chunks).as_retriever(
            search_kwargs={"k": 3}
        )

    return retriever

def create_and_store_embeddings(embedding_model, documents: List[Document]) -> Chroma:
    """Calculates the embeddings and stores them in a Chroma vectorstore."""
    vectorstore = Chroma.from_documents(documents, embedding=embedding_model, collection_name=COLLECTION_NAME, persist_directory=VECTOR_STORE_DIR)
    return vectorstore

def create_rag_chain(embedding_model: JinaEmbeddings, llm: ChatGroq) -> Runnable:
    """Creates the RAG chain."""
    template = """Use provided context to answer the question in smart grid's context. If the prompt is about the model or is it's result, then explain the reason for result and ways to make the system stable, if the result is unstable.
    <context>
    {context}
    </context>

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = get_vectorstore_retriever(embedding_model)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def initialize_system(model_name: str):
    """Initializes and returns the retrieval-augmented generation chain based on the selected model."""
    pdf = load_document()
    chunks = chunk_document(pdf)
    embedding_model = JinaEmbeddings(jina_api_key=environ.get("JINA_API_KEY"), model_name=EMBED_MODEL_NAME)
    vectorstore = create_and_store_embeddings(embedding_model, chunks)
    llm = ChatGroq(temperature=LLM_TEMPERATURE, model_name=model_name)
    return create_rag_chain(embedding_model, llm)

def process_query(chain, query):
    """Processes the query using the initialized RAG chain."""
    return chain.invoke({"input": query})

def main() -> None:
    model_dict={
        'Mixtral':"Mixtral-8x7b-32768",
        'mixtral':"Mixtral-8x7b-32768",
        'Llama':"Llama3-70b-8192",
        'llama':"Llama3-70b-8192",
        'Gemma':"Gemma-7b-it",
        'gemma':"Gemma-7b-it"
    }
    chain = initialize_system(model_dict['gemma'])
    print(process_query(chain, 'What is ANM?')['answer'])


if __name__ == "__main__":
    main()