# 5. RetrievalQAChain: combines a chunk of the document with the most similar embeddings (700-1500 different dimensions) within a vector store,
#  with the user question into a PromptTemplate to be sent to ChatGPT. 
#  TLDR:  User Question + embedding from vector store; packaged in a ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma # opensource vectorstore that saves into sqlite locally; first: pip install chromadb 
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA # the method that 'packages' the user inquiry and the relevant embedding from vector store into a PromptTemplate and sends to CGPT
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever #import the custom retriever class that will auto-filter any documents that are too similar to each other
from dotenv import load_dotenv
import langchain 

langchain.debug=True

load_dotenv()


chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(
    #tells chroma where the SQLite embeddings are stored 
    persist_directory="emb",
    embedding_function=embeddings #creates the embeddings; uses a different name, but same function as in main.py
)



## Retriever: an object that must have a method called "get_relevant_documents" that takes a string and returns a list of documents.
## It helps connect the Chroma db to the RetrievalQA chain, since the developer of RetrievalQA wasn't sure what vector db would be used, the VDB devs had to include that method
## when we call chroma.as_retriever() method, we get the retriever object. The retriever object has the get_relevant_documents method that is basically just calling the Chroma.similarity_search()
# retriever = db.as_retriever() #without duplicate filtration logic
retriever = RedundantFilterRetriever(
    embeddings=embeddings ,
    chroma=db
)


chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    # we are taking the relevant {fact} and the {question} and 'stuffing' them into SystemMessagePromptTemplate and HumanMessagePromptTemplate within the ChatPromptTemplate
    chain_type="stuff")

result = chain.run("What is an interesting fact about the english language?")

print(result)