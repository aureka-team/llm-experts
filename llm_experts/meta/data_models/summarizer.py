from pydantic import BaseModel, StrictStr


class SummarizerExpertInput(BaseModel):
    text: StrictStr
    text_type: StrictStr


class SummarizerExpertOutput(BaseModel):
    summary: StrictStr
