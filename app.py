from fastapi import FastAPI, Query
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
import os

# ✅ Load environment variables (Use Render Environment Variables in production)
GOOGLE_API_KEY = "YOur API" # Replace with your key if testing locally
VECTOR_DB_PATH = "faiss_law_index"  # Path to your saved FAISS index

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# ✅ Load FAISS Vector Database (No need to recreate it)
print("✅ Loading FAISS vector database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
print("✅ FAISS vector database loaded successfully!")

@app.get("/")
def home():
    return {"message": "FAISS RAG API is running!"}

@app.get("/query")
def query_faiss(q: str = Query(..., description="Your search query")):
    """Search FAISS vector store and get the answer from Gemini."""
    docs = vectorstore.similarity_search(q, k=5)  # Retrieve top 5 relevant documents
    retrieved_text = "\n".join([doc.page_content for doc in docs])

    # Use Gemini to generate the answer based on retrieved documents
    response = llm.invoke(f"Answer the following based on the documents:\n{retrieved_text}\n\nQuestion: {q}")
    
    return {"question": q, "answer": response}
