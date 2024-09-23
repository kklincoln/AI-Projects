from .sql_memory import build_memory
from app.chat.memories.window_memory import window_buffer_memory_builder

""" component maps: Dictionaries of retrievers, llms, and one for all memory.
There will be a 'pool' of retrievers, llms, and memory integrations. We will use a randon combination of these,
 allow the user to 'vote' the answer and the answer scoring will factor in to future choices for component combinations """

memory_map = {
    "sql_buffer_memory": build_memory,
    "sql_window_memory": window_buffer_memory_builder
}