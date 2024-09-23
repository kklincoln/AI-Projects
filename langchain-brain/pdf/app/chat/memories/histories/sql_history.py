from pydantic import BaseModel
from langchain.schema import BaseChatMessageHistory

from app.web.api import (
    get_messages_by_conversation_id,    # get all messages tied to a conversation from the SQLite db;
    add_message_to_conversation  # add message tied to a conversation into the SQLite db
)

#  ChatMessageHistory takes messages and stored them in a list. Whenever called, it returns the list.
# we will develop SQLMessageHistory, which is almost a copy of the above, but referencing SQLite instead of a list.
class SqlMessageHistory(BaseChatMessageHistory, BaseModel): #extends two base classes
    conversation_id: str # this is the part within the web chat interface that's shown when you click 'history' dropdown

    @property
    def messages(self):
        # find all messages in database for conversation_id;  conversation_id is selected from web interface
        return get_messages_by_conversation_id(self.conversation_id)

    def add_message(self, message):
        # add message to database
        return add_message_to_conversation(
            conversation_id=self.conversation_id,
            role=message.type,
            content=message.content
        )

    def clear(self):
    # clear database on conversation_id;
    # #not used
        pass