"""rather than having the user wait for the spinning logo while the CGPT is responding to the question, streaming allows
for the user to see the response as it is generated, similar to how it is done within CGPT
Language models are happy to stream, but chains are problematic about streaming, so we need to code accordingly;
 this is overwriting the chat.stream function because the dox say to override for actual effect"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain # to allow us to create a chain out of this streaming process
from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue
from threading import Thread
from dotenv import load_dotenv
load_dotenv()


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue): #use the queue that is passed in at the time of calling streaminghandler (inside StreamingChain)
        self.queue = queue

    """'handles' the ChatOpenAI() part of our program to actually stream the output as it is received from OpenAI servers"""
    def on_llm_new_token(self, token, **kwargs):
        """ when OpenAI Servers send us a new response in the form of a token, on_llm_new_token, put the token in the queue;
        then a generator inside the 'stream' function will use a while loop to scan the queue for elements, and will then
        show up within the yield statements in StreamingChain"""
        self.queue.put(token)

        """When there are no more anticipated responses from the OpenAI Servers, add a value of 'None' to the queue"""
    def on_llm_end(self, response, **kwargs):
        self.queue.put(None)

    def on_llm_error(self, error, **kwargs):
        self.queue.put(None)



chat = ChatOpenAI(
    streaming=True #streaming forces OpenAI to responds to LangChain within our app as streamed response
    )
prompt = ChatPromptTemplate.from_messages([
    # put in a list that has a tuple that has "type of message":"actual message template"
    ("human", "{content}")
])


class StreamableChain: #this is overwriting the chat.stream function because the dox say to override for actual effect
    """ makes the chain gather the output from the StreamingHandler  """
    def stream(self, input): #input will be a dictionary that's the {"content":"tell me a joke"} part of the chain; humanmessage
        """ create a new queue and handler, so that every user within the site will have access to their own; scalability"""
        queue = Queue()
        handler = StreamingHandler(queue)

        def task(): #executing the chain (format prompt, send it to LLM, LLM sends to OpenAI servers); use handler for this call
            self(input, callbacks=[handler])

        """ tells a new thread to start, allowing the chain to run concurrently and then continue the rest of StreamingChain
        rather than having to wait for the self(input) to be finished, which is calling the LLMChain(input). makes the streaming work"""
        Thread(target=task).start()

        while True:
            token = queue.get() #look in the queue to get anything that is inside the queue
            if token is None:
                break
            yield token # yield the token from the queue

class StreamingChain(StreamableChain, LLMChain): #using 'StreamableChain' mixin allows to add support to the LLMChain
    pass

chain = StreamingChain(llm=chat, prompt=prompt)
for output in chain.stream(input={"content": "Tell me a joke"}):
    print(output)