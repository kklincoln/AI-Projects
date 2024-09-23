from langchain.callbacks.base import BaseCallbackHandler

"""rather than having the user wait for the spinning logo while the CGPT is responding to the question, streaming allows
for the user to see the response as it is generated, similar to how it is done within CGPT
Language models are happy to stream, but chains are problematic about streaming, so we need to code accordingly;
 this is overwriting the chat.stream function because the dox say to override for actual effect"""


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue): #use the queue that is passed in at the time of calling streaminghandler (inside StreamingChain)
        self.queue = queue
        self.streaming_run_ids = set() #keeps track of all of the run_ids that are inside a streaming model

    def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        if serialized["kwargs"]["streaming"]: #the run_id that is tied to a streaming model
            self.streaming_run_ids.add(run_id) #adds the run_id that is inside this streaming model to allow for adding 'none' to queue

    """'handles' the ChatOpenAI() part of our program to actually stream the output as it is received from OpenAI servers"""
    def on_llm_new_token(self, token, **kwargs):
        """ when OpenAI Servers send us a new response in the form of a token, on_llm_new_token, put the token in the queue;
        then a generator inside the 'stream' function will use a while loop to scan the queue for elements, and will then
        show up within the yield statements in StreamingChain"""
        self.queue.put(token)

        """When there are no more anticipated responses from the OpenAI Servers, add a value of 'None' to the queue"""
    def on_llm_end(self, response, run_id, **kwargs):
        if run_id in self.streaming_run_ids: #if the llm_end is called by a streaming model,  put None in queue
            self.queue.put(None)
            self.streaming_run_ids.remove(run_id) #remove the run_id because now it's not needed

    def on_llm_error(self, error, **kwargs):
        self.queue.put(None)


