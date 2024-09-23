from functools import partial
from .pinecone import  build_retriever

""" component maps: Dictionaries of retrievers, llms, and one for all memory.
There will be a 'pool' of retrievers, llms, and memory integrations. We will use a randon combination of these,
 allow the user to 'vote' the answer and the answer scoring will factor in to future choices for component combinations """

retriever_map = {
    "pinecone_1": partial(build_retriever, k=1),
    "pinecone_2": partial(build_retriever, k=2),
    "pinecone_3": partial(build_retriever, k=3),
}
