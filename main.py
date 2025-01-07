import os
import openai
import gradio as gr
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Step 1: Load and Preprocess Data
def preprocess_documents(file_path):
    """
    Load and split large documents into smaller chunks.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(documents)

# Step 2: Generate Embeddings and Build the Vector Database
def create_vector_store(documents):
    """
    Create a vector database from documents using OpenAI embeddings.
    """
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

# Step 3: Save and Load Vector Database
def save_vector_db(vector_db, file_path):
    vector_db.save_local(file_path)

def load_vector_db(file_path):
    return FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Step 4: Retrieve Relevant Documents
def retrieve_docs(query, vector_db, top_k=3):
    """
    Retrieve the top_k relevant documents for a query.
    """
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# Step 5: Generate GPT-4 Response
def generate_response(query, context):
    """
    Generate a GPT-4 response given a query and context.
    """
    prompt = f"""
    Use the following context to answer the question:
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"].strip()

# Step 6: Gradio UI for RAG Application
def rag_app(query):
    """
    Full RAG pipeline: Retrieve context and generate response.
    """
    context = retrieve_docs(query, vector_db)
    response = generate_response(query, context)
    return f"**Answer:**\n{response}\n\n**Context Used:**\n{context}"

# --- Main Program ---
if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # Load or create vector database
    VECTOR_DB_PATH = "vector_store"
    DOCUMENT_PATH = "./data.txt"

    if os.path.exists(VECTOR_DB_PATH):
        vector_db = load_vector_db(VECTOR_DB_PATH)
    else:
        print("Creating vector database...")
        docs = preprocess_documents(DOCUMENT_PATH)
        vector_db = create_vector_store(docs)
        save_vector_db(vector_db, VECTOR_DB_PATH)
        print("Vector database created and saved!")

    # Gradio Interface
    iface = gr.Interface(
        fn=rag_app,
        inputs="text",
        outputs="markdown",
        title="RAG Application with GPT-4",
        description="Ask questions and get responses augmented with relevant context.",
        examples=["What is the capital of France?", "Explain the theory of relativity."],
        flagging_mode="never"
    )

    # Launch Gradio App
    iface.launch(share=False)
