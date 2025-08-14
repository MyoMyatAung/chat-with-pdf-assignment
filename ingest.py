import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def ingest_pdfs(pdf_folder: str, index_path: str = "faiss_index"):
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs = loader.load()
            documents.extend(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"Ingested {len(chunks)} chunks into FAISS at {index_path}")

if __name__ == "__main__":
    ingest_pdfs("./papers/")
