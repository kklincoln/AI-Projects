from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
#Sequential chain allows us to pass the two chains in a sequential manner
from langchain.chains import LLMChain, SequentialChain

import argparse #allowing us to pass in our arguments for language and task as command line inputs rather than hard-coded values
from dotenv import load_dotenv
import os

load_dotenv()

parser = argparse.ArgumentParser()
# establish the parser argument expectations from the CLI, expects --task and --language, use defaults if not provided
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args() # args will then parse the CLI for the arguments and use the defaults if not provided


llm = OpenAI() #this looks for a very specifically named environment variable: OPENAI_API_KEY


# this is the prompt template that we will feed to the llm; when we pass in the input_variables associated with the language and task, the prewritten
# template tells the model what we want it to do, structured in a way to achieve the desired outcome. 
code_prompt = PromptTemplate(
    template = "Write a very short {language} function that will {task}.",
    input_variables = ['language','task']
)

#second prompt template: to be used to feed the output from the first chain into the second llm call
test_prompt = PromptTemplate(
    input_variables=["language","code"],
        template="Write a test for the following {language} code:\n{code}"
)


# LLMChains need to have arguments associated with what llm to use as well as what to prompt it with (in this case, the template above: code_prompt)
code_chain = LLMChain(
    llm=llm, #the OpenAI() instance above
    prompt=code_prompt,
    output_key="code" #passing this additional configuration so that the default "text" output is renamed more appropriately for input into the next chain as "code"
)
# Sequential Action: test chain this is the chain that is executed second, dependent on outputs from the first 
test_chain = LLMChain(
    llm=llm, #still using the instance of the OpenAI class from above
    prompt=test_prompt, 
    output_key="test" #passing this additional configuration so that the default "text" output is renamed more appropriately for input into the next chain as "test"
)



#SequentialChain is the chain combo 
chain = SequentialChain(
    #pass in the chains that we want to run one after another
    chains=[code_chain, test_chain],
    #pass in the only human input variables associated with our calls to the llm
    input_variables=["language","task"],
    #the output variables that we are concerned with. take the code from chain1 and test from chain2
    output_variables=["code","test"]
    )



# provide a dictionary that contains all the input variables we need to get this to run (the ones from the code_prompt template: language and task)
result = chain({
    "language": args.language,
    "task": args.task
})

# print(result) #result is a dictionary that contains all the inputs as well as the outputs. The output is associated with a key: 'text'
print(">>>>> GENERATED CODE:")
print(result["code"]) #this generates the output code without the full dictionary
print(">>>>> GENERATED TEST:")
print(result["test"])