from flask import current_app
from queue import Queue
from threading import Thread
from app.chat.callbacks.stream import StreamingHandler

""" effectively, this StreamableChain class can be added to any of the other chains from langchain out of the box, to allow
for a streaming response as we have generated here, so long as you pass it on instantiation, as demonstrated in retrieval.py"""

class StreamableChain: #this is overwriting the chat.stream function because the dox say to override for actual effect
    """ makes the chain gather the output from the StreamingHandler  """
    def stream(self, input): #input will be a dictionary that's the {"content":"tell me a joke"} part of the chain; humanmessage
        """ create a new queue and handler, so that every user within the site will have access to their own; scalability"""
        queue = Queue()
        handler = StreamingHandler(queue)

        def task(app_context): #executing the chain (format prompt, send it to LLM, LLM sends to OpenAI servers); use handler for this call
            app_context.push() # gives access to current_user, database, etc.; this needs to be added because we are using Flask
            self(input, callbacks=[handler])

        """ tells a new thread to start, allowing the chain to run concurrently and then continue the rest of StreamingChain
        rather than having to wait for the self(input) to be finished, which is calling the LLMChain(input). makes the streaming work"""
        Thread(target=task, args=[current_app.app_context() ]).start() #args:

        while True:
            token = queue.get() #look in the queue to get anything that is inside the queue
            if token is None:
                break
            yield token # yield the token from the queue
