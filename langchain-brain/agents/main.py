# -What is an agent? Almost identical to a chain, but an agent:
# 		- knows how to use tools	(the only difference)
# 		- will take that list of tools and convert them into JSON function descriptions
# 		- still has input variables, memory, prompts, etc- same as a normal chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import(
    ChatPromptTemplate, 
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from tools.sql import run_query_tool, list_tables, describe_tables_tool # tool we created that CGPT uses to run queries
from tools.report import write_report_tool #tool that builds HTML report of the output
from handlers.chat_model_start_handler import ChatModelStartHandler


load_dotenv() 


#initialize the class that prints the messages on chat model start; great for debugging. 
handler = ChatModelStartHandler()
# create chat model
chat = ChatOpenAI(callbacks=[handler])  

tables = list_tables()
# create chat prompt template
prompt = ChatPromptTemplate(
        messages =[
            SystemMessage(content=(
                "You are an AI that has access to a SQLite Database.\n"
                f"The database has tables of: {tables}\n"
                "Do not make any assumptions about what tables exist or what columns exist."
                "instead, use the 'describe_tables' function"
            )),
            MessagesPlaceholder(variable_name="chat_history"), # add an expected variable called "chat_history" which is a ConversationBufferMemory from previous interactions
            HumanMessagePromptTemplate.from_template("{input}"),#assume the input variable with have name "input" 
            #agent_scratchpad: simplified form of memory; ~captures the intermediate messages(e.g. calling the query and sending result back to cGPT; AssistantMessage  & FunctionMessage)               
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
    )



# in order for a 'memory' to be stored for this prompt, we have to implement some type of memory. 
#NOTE: when using memory, CGPT loses the intermediate steps and only saves the HumanMessage and the AIMessage (start and finish of convo)
memory= ConversationBufferMemory(memory_key="chat_history", return_messages=True) # r_m=True means give us the messages as message objects rather than strings
#list of tools created for cGPT to be able to use
tools = [run_query_tool, describe_tables_tool, write_report_tool]

#create an agent with the chat model, to execute the tool of querying the db. Same thing as a chain, but knows how to use tools.
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

#It's a simple object that takes an agent (chain) and runs it over and over until the response from CGPT is *not* a function call. Basically a While loop
agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory
)


agent_executor("How many orders are there?")
agent_executor("Repeat the exact same process for users.")