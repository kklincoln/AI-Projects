from langchain.embeddings.base import Embeddings 
from langchain.vectorstores import Chroma
#need to import this to be able to create a CustomFilterRetriever; defines what a retriever is and how it behaves, needs to have 
from langchain.schema import BaseRetriever # needs to have get_relevant_documents(self, query)



#  Retriever: an object that must have a method called "get_relevant_documents" that takes a string and returns a list of documents. It helps connect the
#  Chroma db to the RetrievalQA chain, since the developer of RetrievalQA wasn't sure what vector db would be used, the VDB devs have to include that method.
#  When we call chroma.as_retriever() method, we get the retriever object. The retriever object has the get_relevant_documents method that is basically
#  just calling the Chroma.similarity_search()
class RedundantFilterRetriever(BaseRetriever):
    # rather than hardcoding embeddings and chroma, we require these to be provided of type Embeddings and Chroma respectively, in case the instance names change
    embeddings: Embeddings # embedding object of type Embeddings; must be provided to calculate embeddings 
    chroma: Chroma # must provide an already initialized instance of Chroma

    def get_relevant_documents(self, query): #any retriever needs to have this method
        #calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)

        #take embeddings and feed them into the 
        # max_marginal_relevance_search_by_vector  method that automatically removes duplicates
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb , # the embed_query established above is what we are considering similarities against
            lambda_mult = 0.8 # the closer this number is to 1, the more similar the documents can be.
            )
    
    #this one won't be used for this project, but needs to be established
    async def aget_relevant_documents(self):
        return []