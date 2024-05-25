from pydantic import BaseModel, StrictStr


class GenericAssistantInput(BaseModel):
    user_query: StrictStr


class GenericAssistantOutput(BaseModel):
    assistant_response: StrictStr
