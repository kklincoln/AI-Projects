# CONTEXT:
# // most llms follow a 'completion style' of text generation. if we have input that is an incomplete sentence,the model will make 
    # a prediction of the remainder of the sentence given the context. The inputs and outputs are simple and typically text types.
# // the other type we make use of is a 'Conversation style' model like ChatGPT, the inputs are a bit more vague, you have to distinguish between user and system messages
    # OpenAI message_types: user, system(optional that customizes how the chat bot behaves and responds), assistant (created by chat model)
    # LangChain Message_types: Human, System, AI
    #  system: "You are a chatbot specializing in {subject}"
    # NOTE: when you message CGPT, it sends the whole message history each time you continue the conversation, it doesn't "remember" the flow of the convo. 
    # this is why it's important to be concise and start new chats when possible
from langchain.chat_models import ChatOpenAI    #langchain thinks that llm is a completion model, so it isn't packaged in langchain.llms as previously done
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# initializes the ChatOpenAI class which is establishing the connection to openAI server
chat = ChatOpenAI(verbose=True) #this searches the .env for the OPENAI_API_KEY to then use that permission to call upon the llm 

memory = ConversationSummaryMemory(
    # chat_memory = FileChatMessageHistory("messages.json"), USE THIS ONLY WITH INSTANCE OF ConversationBufferMemory # stores a json associated with the conversation into messages.json
    memory_key="messages", #when we feed our input into the chain, we can feed in additional keys into the chain, "messages" is a compilation of previous outputs
    return_messages=True, # this makes sure that the objects are returned for the "messages" value pair, rather than str, so that we can see which sent it (user,ai,system)
    llm=chat #tells the chain to reference the ChatOpenAI llm when it calls to summarize the conversation with the PromptTemplate
    ) 

# the prompt that is sent off to LLM: ChatGPT
prompt = ChatPromptTemplate(
    # the variables that we pass through to CGPT: "content": what message we ask; "messages": if conversation history, include it
    input_variables=["content","messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"), #this is the placeholder instance that tells our promptTemplate to look for a variable called "messages"
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

#create a chain that wires the chat model and the prompt together
chain = LLMChain(
    llm=chat,
    prompt=prompt, # the prompt is whatever is passed in by the user as {content}
    memory=memory,  #ensure that the chain has access to the memory 
    verbose=True #adds the debugging so you can see that the conversationSummaryMemory is being created
)


while True:
#receive user input ; enter key triggers action
    content = input(">> ")

    #in order to run the chain and get 'result' we have to provide a dictionary with all of the input variables that the chain requires. 
    # in this case, just the content the user passed in above
    result = chain({"content": content})

    print(result['text'])  #default output key is called 'text'
