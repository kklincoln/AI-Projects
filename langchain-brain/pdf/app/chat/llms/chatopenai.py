from langchain.chat_models import ChatOpenAI

#define and create an instance of the language model. ChatOpenAI as previously used.
def build_llm(chat_args, model_name):
    return ChatOpenAI(
        streaming=chat_args.streaming
        ,model_name=model_name
    )