from functools import partial
from .chatopenai import build_llm

""" component maps: Dictionaries of retrievers, llms, and one for all memory.
There will be a 'pool' of retrievers, llms, and memory integrations. We will use a randon combination of these,
 allow the user to 'vote' the answer and the answer scoring will factor in to future choices for component combinations """

llm_map={
    #when we call partial, we run the original build_llm function with the kwargs we define next to it
    "gpt-4": partial(build_llm, model_name="gpt-4"),
    "gpt-3.5-turbo": partial(build_llm, model_name="gpt-3.5-turbo")
}