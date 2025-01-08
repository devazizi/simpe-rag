import os
import openai
import gradio as gr
from langchain.vectorstores import FAISS
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np
import requests
import json


def set_openai_api_base(api_type):
    if api_type == "embedding":
        openai.api_base = os.getenv('EMBEDDING_API_URL')
        openai.api_key = os.getenv('EMBEDDING_API_KEY')
    elif api_type == "chat":
        openai.api_key = os.getenv('OPENAI_API_KEY')
        print(openai.api_key)
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def preprocess_documents(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_documents(documents)


def generate_embeddings(texts):
    set_openai_api_base("embedding")
    response = openai.Embedding.create(
        model="BAAI/bge-m3",  # Replace with your model
        input=texts
    )
    embeddings = [item["embedding"] for item in response["data"]]
    return embeddings


def index_to_docstore_id(index_id):
    return index_id  # Assuming the document ID is the same as the index ID


def create_vector_store(documents):
    texts = [doc.page_content for doc in documents]
    embeddings = generate_embeddings(texts)

    embeddings = np.array(embeddings).astype('float32')
    embedding_dim = len(embeddings[0])
    print(f"Embedding Dimension: {embedding_dim}")

    index = faiss.IndexFlatL2(embedding_dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    docstore = {i: doc for i, doc in enumerate(documents)}

    vector_db = FAISS(
        index=index,
        embedding_function=generate_embeddings,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id  # Use the global function
    )
    return vector_db


def save_vector_db(vector_db, file_path):
    vector_db.save_local(file_path)


def load_vector_db(file_path):
    return FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


def retrieve_docs(query, vector_db, top_k=2):
    query_embedding = generate_embeddings([query])  # Generate embedding for the query
    query_embedding = np.array(query_embedding, dtype='float32')

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    D, I = vector_db.index.search(query_embedding, top_k)  # D is distances, I is indices

    docs = [vector_db.docstore[i] for i in I[0]]
    return "\n\n".join([doc.page_content for doc in docs])


#
# def generate_response(query, context):
#
#     set_openai_api_base('chat')
#     prompt = f"""
#     Use the following context to answer the question with  make clean text:
#
#     Context:
#     {context}
#
#     Question:
#     {query}
#
#     Answer:
#     """
#     import requests
#     try:
#         url = "https://api.openai.com/v1/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
#             "Content-Type": "application/json",
#         }
#         data = {
#             "model": "gpt-4",
#             "messages": [
#                 {"role": "system", "content": context},
#                 {"role": "user", "content": query},
#             ],
#         }
#         response = requests.post(url, json=data, headers=headers)
#         print("Raw response:", response.text)  # Log raw response
#         response.raise_for_status()
#         yield response.json()["choices"][0]["message"]["content"].strip()
#         # response = openai.ChatCompletion.create(
#         #     model="gpt-4",
#         #     messages=[
#         #         {"role": "system", "content": context},
#         #         {"role": "user", "content": query}
#         #     ]
#         # )
#         # return response["choices"][0]["message"]["content"].strip()
#     except openai.error.OpenAIError as e:
#         print("OpenAI API error:", e)
#
#         yield "unexpected error"
#     except Exception as e:
#         print("Unexpected error:", e)
#
#         yield "unexpected error"
#
#     # response = openai.ChatCompletion.create(
#     #     model="gpt-4",
#     #     messages=[
#     #         {"role": "system", "content": "You are a helpful assistant."},
#     #         {"role": "user", "content": prompt}
#     #     ],
#     #     max_tokens=300
#     # )
#     # return
#

def generate_response(query, context):
    set_openai_api_base("chat")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"""
    Use the following context to answer the question with clean text:

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = openai.ChatCompletion().create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
    )

    return response["choices"][0]["message"]["content"].strip()


def rag_app(query, history):
    context = retrieve_docs(str(query), vector_db)

    # print(context)
    yield generate_response(query, context)

    yield f"from context: {context}"
    #
    # return f"**Answer:**\n{response}\n\n**Context Used:**\n{context}"


if __name__ == "__main__":
    load_dotenv()
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


    def slow_echo(message, history):
        for i in range(len(message)):
            import time
            time.sleep(0.3)
            yield "You typed: " + message[: i + 1]


    iface = gr.ChatInterface(
        fn=rag_app,
        type="messages",
    )

    #
    # iface = gr.Interface(
    #     fn=rag_app,
    #     inputs="text",
    #     outputs="markdown",
    #     title="RAG Application with GPT-4",
    #     description="Ask questions and get responses augmented with relevant context.",
    #     examples=["What is the capital of France?", "Explain the theory of relativity."],
    #     flagging_mode="never"
    # )

    iface.launch(share=False)
