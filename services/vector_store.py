from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 2. Initialize the local embedding model
embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def create_retriever(documents):
    """
    Create a vector store and retriever from loaded documents.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        A retriever for the supplied documents.
    """

    if not documents:
        raise ValueError("No readable content was found in the uploaded files.")

    # 1. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Could not create any text chunks from the documents.")

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    

    # 3. Create a vector store for this document set
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name="paperpilot_uploads",
    )

    # 4. Create the retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10
        }
    )

    return retriever