from pydantic import BaseModel, StrictStr


class SummarizerInput(BaseModel):
    text: StrictStr
    text_type: StrictStr


class SummarizerOutput(BaseModel):
    summary: StrictStr
