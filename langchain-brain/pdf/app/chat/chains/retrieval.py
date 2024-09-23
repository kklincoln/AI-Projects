from langchain.chains import ConversationalRetrievalChain
from app.chat.chains.streamable import StreamableChain
from app.chat.chains.traceable import TraceableChain


#this is a custom class that is just essentially combining ConversationalRetrievalChain that also supports StreamableChain
class StreamingConversationalRetrievalChain(TraceableChain, StreamableChain, ConversationalRetrievalChain):
    pass