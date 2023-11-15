from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import pinecone
from langchain.document_loaders import WebBaseLoader


# Function to fetch the data from the website
def get_website_data(website):
    from langchain.document_loaders import UnstructuredHTMLLoader

    loader = UnstructuredHTMLLoader(website)
    data = loader.load()
    return data


# Function to split the data into smaller chunks
def split_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )

    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks


# Function to create embeddings instance
def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    return embeddings


# Pushing the embeddings to pinecone data store
def push_to_pinecone(
    pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs
):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)
    index_name = pinecone_index_name
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index


# Function to pull index data from pinecone
def pull_from_pinecone(
    pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings
):
    pinecone.init(api_key=pinecone_apikey, environment=pinecone_environment)

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


# This function will help us in fetching the top relevant documents from our vector store- Pinecone Index
def get_similar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


get_website_data("./Router.html")
