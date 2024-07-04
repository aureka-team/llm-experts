from pydantic import BaseModel, PositiveInt, StrictStr


class TextChunk(BaseModel):
    chunk_id: PositiveInt
    text: StrictStr


class RretrieverFilterInput(BaseModel):
    text_chunks: list[TextChunk]
    query_text: StrictStr


class RretrieverFilterInputWrapper(RretrieverFilterInput):
    text_chunks: StrictStr


class RretrieverFilterOutput(BaseModel):
    used_chunk_ids: list[PositiveInt] = []
