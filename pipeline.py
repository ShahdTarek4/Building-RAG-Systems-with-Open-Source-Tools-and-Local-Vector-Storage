from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def load_documents(directory='documents'):
    documents = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        try:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(path)
            elif file.endswith('.txt'):
                loader = TextLoader(path)
            else:
                print(f"Unsupported format: {file}")
                continue
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file  
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    print(f"📄 Loaded {len(documents)} documents.")
    return documents



def split_documents(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    print(f"✂️ Split into {len(split_docs)} chunks.")
    return split_docs


def embed_and_store(chunks, model_name='all-MiniLM-L6-v2', save_path='faiss_store'):
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(save_path)
    print(f"Vector store saved to {save_path}")