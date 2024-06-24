import os

from pymongo import MongoClient
from llm_experts.meta import StoreMessages


MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017",
)


class MongoStore:
    def __init__(
        self,
        mongo_uri=MONGO_URI,
        mongo_database="llm-experts",
    ):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[mongo_database]

    def __del__(self) -> None:
        self.client.close()

    def save_message_history(
        self,
        store_messages: StoreMessages,
        collection: str = "message-histories",
    ) -> None:
        doc = store_messages.model_dump()
        self.db[collection].update_one(
            {"_id": doc["session_id"]},
            {"$set": {"messages": doc["messages"]}},
            upsert=True,
        )
