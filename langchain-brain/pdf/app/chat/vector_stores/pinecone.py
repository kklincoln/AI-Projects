import os
import pinecone #pinecone client
from langchain.vectorstores.pinecone import Pinecone
from app.chat.embeddings.openai import embeddings

#initialize pinecone client
pinecone.Pinecone(
    api_key = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENV_NAME"), # Region of globe where environment is hosted
)

#initialize vector store; basically a langchain wrapper that is the "db" variable
vector_store = Pinecone.from_existing_index(    # already created index in the pinecone.io dashboard
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

# # only look inside pinecone 'db' for documents that align with the arguments for pdf_id
def build_retriever(chat_args, k):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}, #customizes how pinecone looks up documents inside our vectorstore
                    "k": k #specifies how many documents we want to have returned; defualt 1
    }
    return vector_store.as_retriever(
        search_kwargs=search_kwargs
    )
