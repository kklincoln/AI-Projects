import random
from langfuse.model import CreateTrace
from app.chat.tracing.langfuse import langfuse

class TraceableChain:
    def __call__(self, *args, **kwargs):
        # create callbacks handler
        trace = langfuse.trace(  # tracing.langfuse file is where the tracing can be viewed
            CreateTrace(
                id=self.metadata["conversation_id"],
                # on langfuse platform, when run this chain, take any info & append to this conv_id
                metadata=self.metadata
            )
        )
        # add into list of callbacks
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(trace.getNewHandler())
        kwargs["callbacks"] = callbacks

        # pass that into the list of callback execution of the chain.
        return super().__call__(*args, **kwargs)