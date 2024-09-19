from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory


def get_mongodb_chat_history(
    session_id: str,
    collection_name: str,
    database_name: str = "llm-experts",
    connection_string: str = "mongodb://localhost:27017",
) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=connection_string,
        database_name=database_name,
        collection_name=collection_name,
    )
