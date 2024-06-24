from pydantic import BaseModel, StrictStr, PositiveInt


class ResponseMetadata(BaseModel):
    model: StrictStr
    completion_tokens: PositiveInt
    prompt_tokens: PositiveInt
    total_tokens: PositiveInt


class StoreMessage(BaseModel):
    human_message: StrictStr
    ai_message: dict
    response_metadata: ResponseMetadata


class StoreMessages(BaseModel):
    session_id: StrictStr
    messages: list[StoreMessage]
