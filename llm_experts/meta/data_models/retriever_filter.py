from pydantic import BaseModel, PositiveInt, StrictStr


class TextChunk(BaseModel):
    chunk_id: PositiveInt
    text: StrictStr


class RretrieverFilterExpertInput(BaseModel):
    text_chunks: list[TextChunk]
    query_text: StrictStr


class RretrieverFilterExpertInputWrapper(BaseModel):
    text_chunks: StrictStr
    query_text: StrictStr


class RretrieverFilterExpertOutput(BaseModel):
    used_chunk_ids: list[PositiveInt] = []
