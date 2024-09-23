from langchain.memory import ConversationBufferMemory
from app.chat.memories.histories.sql_history import SqlMessageHistory

# build memory, called with chat_args object
def build_memory(chat_args):
    #override the default chat memory with an instance of SqlMessageHistory
    return ConversationBufferMemory(
        chat_memory=SqlMessageHistory(
            conversation_id=chat_args.conversation_id
        ),
        return_messages=True,#returns as message object instead of string
        #in a conversationalQA chain, input variables include {"question":"", "chat_history":""}
        #when it outputs, the output variables object has key of {"answer":""}
        # whenever build_memory is called, it should pass all of the chat_history into the input variables and gather the output "answer"
        memory_key="chat_history",
        output_key="answer"
    )
