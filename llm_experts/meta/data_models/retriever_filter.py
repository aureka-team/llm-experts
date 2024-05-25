from pydantic import BaseModel, PositiveInt, StrictStr


class TextChunk(BaseModel):
    chunk_id: PositiveInt
    text: StrictStr


class RretrieverFilterInput(BaseModel):
    text_chunks: list[TextChunk]
    query_text: StrictStr


class RretrieverFilterInputWrapper(BaseModel):
    text_chunks: StrictStr
    query_text: StrictStr


class RretrieverFilterOutput(BaseModel):
    used_chunk_ids: list[PositiveInt] = []
