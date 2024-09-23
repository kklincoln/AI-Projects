from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen   #for the callback handler "wrap" the messages by the applicable senders in colored textboxes


# create a function that takes in arguments and any number of kwargs and prints results of passing in those arguments
def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))
    


class ChatModelStartHandler(BaseCallbackHandler):
    # create a new callback class that is triggered by the event on_chat_model_start
    def on_chat_model_start(self, serialized, messages, **kwargs):     # messages type: List[List[BaseMessage]]
        print("\n\n\n\n=========== Sending Messages =============\n\n")
        for message in messages[0]:
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")
            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")
            #ai attempting a function call
            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"] #if there is a function call, pull out the function being called
                boxen_print(f"running tool {call['name']} with args {call['arguments']}",
                            title=message.type, 
                            color="cyan")
            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")
            # us sending the result of a function call
            elif message.type == "function":
                boxen_print(message.content, title=message.type, color="purple")
            else: 
                boxen_print(message.content, title=message.type)


