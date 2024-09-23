#context: These documentloaders will let you load a file and return with a 'Document'.
#  The 'Document' inside of langchain is a dictionary that has 
# {"page_content":"...", "metadata":"{"source":"facts.txt"}"}

# squaredL2 similarity is the distance between two points (two vectors) to figure out how similar they are
# Cosine similarity is using the angle between the two vectors to determine how similar they are

from langchain.document_loaders import TextLoader #also has PyPDFLoader, JSONLoader, UnstructuredMarkdownLoader (.md files), S3FileLoader (Amazon S3 stores)
from langchain.text_splitter import CharacterTextSplitter # for the 'chunk' creation from the text file 
from langchain.embeddings import OpenAIEmbeddings # OpenAI embeddings generation for 1536 embeddings 
from langchain.vectorstores.chroma import Chroma # opensource vectorstore that saves into sqlite locally; first: pip install chromadb 
from dotenv import load_dotenv

load_dotenv()


#-------------------------------- INSTANTIATE EMBEDDING GENERATOR
embeddings = OpenAIEmbeddings() #the return object has a function embeddings.embed_query(str) that generates 1536 embeddings; 


#-------------------------------- INSTANTIATE TEXT SPLITTER
#  parses the loaded file, count chunk_size, then find nearest separator to cut a 'chunk'
text_splitter = CharacterTextSplitter(
    separator="\n", # attempt to split the text based upon any new line character
    chunk_size=200, # amount of characters
    chunk_overlap=100 #have some similarities in endpoints so that any relevant context by surrounding items can be proviced if possible.  
)


#-------------------------------- LOAD OUR FACTS FILE
loader = TextLoader("facts.txt")
#-------------------------------- SPLIT THE TEXT INTO CHUNKS
docs = loader.load_and_split( #tells the loader to go and load the information from the facts.txt file, similar to pd.read_csv("filename"), stores it as 'Document' objects
    text_splitter=text_splitter
)

#-------------------------------- INITIALIZE DB, CALCULATE EMBEDDINGS, AND  STORE EMBEDDINGS IN VECTOR STORE
 # create a chroma instance and immediately call OPENAI to calculate embeddings for all the documents within 'docs' variable; costs money. 
#  NOTE: This will insert duplicates each re-run
db = Chroma.from_documents(
    docs, 
    embedding=embeddings, # OpenAIEmbeddings process, note the naming convention change from embeddings (above) to embedding.
    persist_directory="emb" #this is the local db storing the embeddings
)

results = db.similarity_search(
    "What is an interesting fact about the English language?",
    k=3 #only return the # most similar document(s); default is 4
    )

# return the Documents that are associated most similarly to the embeddings from the question in 'results' variable 
for result in results:
    print("\n") 
    # print(result[1])    # print the "search score" output from llm; needs above to use: results=db.similarity_search_with_score()
    print(result.page_content) # print the page_content from the document


#-------------------------------- CREATE EMBEDDING OUT OF THE USER'S QUESTION 
#this is done in prompt.py; User's question is combined with the {context} within a template and then sent to the llm

#-------------------------------- DO EMBEDDING SIMILARITY SEARCH
# find relevant facts with Semantic Search. if the user phrases the question in an odd way, we use embeddings to test the similarity of the embeddings
# embeddings are a list of numbers between -1 and 1 that show how much a piece of text is talking about some particular predefined quality (we don't know the associated scale)

# see db.similarity_search_by_vector(emb)

# max_marginal_relevance_search_by_vector() # removes duplicates automatically

#-------------------------------- COMBINE USER QUESTION AND THE RELEVANT DOCUMENT TEXT INTO PROMPTTEMPLATE FOR CHATGPT


# print(docs)