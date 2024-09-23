from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.memories import memory_map
from app.chat.llms import llm_map
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from app.web.api import (get_conversation_components, set_conversation_components) # for use with the random component combinations
from app.chat.score import random_component_by_score



def select_component(component_type, component_map, chat_args):
    components = get_conversation_components(chat_args.conversation_id)#pass in convo id to get the associated components
    previous_component = components[component_type] #with arg component_type (llm,memory,retriever) pull out components
    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args) #return the name of the component and the built component itself
    else:
        random_name = random_component_by_score(component_type, component_map) # select a random key from the component_map
        builder = component_map[random_name]    # build the component for the associated random key
        return random_name, builder(chat_args)  # return component name and the built component

# this will build and return a chain that will be used by the app.web module.
def build_chat(chat_args: ChatArgs):
    # building the chat function that is called, triggers a retriever call from pinecone.py, returning conversations
    # filtered by chat_args.pdf_id
    """ if first message of conversation, get random combination of convo components for use, store them linked to convo_id"""
    retriever_name, retriever = select_component("retriever", retriever_map, chat_args)
    llm_name, llm = select_component("llm", llm_map, chat_args) #note, the chat_args.streaming function is passed in here
    memory_name, memory = select_component("memory", memory_map, chat_args) #this memory allows for sql_memory.py memory context from the sqlite db re: conversation_id

    print(f"Running chain with memory:{memory_name}, llm: {llm_name}, and retriever: {retriever_name}.")
    """set the conversation components for future use"""
    set_conversation_components(chat_args.conversation_id,
                                llm=llm_name,
                                retriever=retriever_name,
                                memory=memory_name)
    condense_question_llm = ChatOpenAI(streaming=False) # this should start a second Retrieval Model that is specifically assigned to the: condense question chai

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm= condense_question_llm,
        memory=memory,
        retriever=retriever,
        metadata=chat_args.metadata
    )
