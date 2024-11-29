from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import streamlit as st
import numpy as np


# Connect to MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/Test")
db = mongo_client["rag_for_chatbot"] 
collection = db["documents"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  

def extract_text_from_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    else:
        st.error("Unsupported file type!", icon="🚨")
        return ""

def generate_text_embeddings(uploaded_files):
    """
    Generates embeddings for the extracted content
"""
    user_uploaded_docs = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            text_content = extract_text_from_file(uploaded_file)
            if text_content:
                embedding = embedding_model.encode(text_content).tolist()
                user_uploaded_docs.append({
                    "filename": uploaded_file.name,
                    "text": text_content,
                    "embedding": embedding,
                })
    return user_uploaded_docs

def search_user_uploaded_docs(query, user_docs, top_k=1):
    """
    Searches for the most relevant content in the user's uploaded files.
"""

    query_embedding = embedding_model.encode(query).tolist()

    def cosine_similarity(vec1, vec2):
        """
            Compute similarity manually (cosine similarity)
    """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    results = [
        {"text": doc["text"], "score": cosine_similarity(query_embedding, doc["embedding"])}
        for doc in user_docs
    ]
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return [res["text"] for res in sorted_results[:top_k]]



