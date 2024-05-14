from typing import Optional
from pydantic import BaseModel, StrictStr, PositiveInt, field_validator

from .validators import languge_validator


class TextChunk(BaseModel):
    chunk_id: PositiveInt
    text: StrictStr


class RAGExpertInput(BaseModel):
    text_chunks: list[TextChunk]
    query_text: StrictStr
    output_language: StrictStr

    _source_language_validator = field_validator("output_language")(
        languge_validator
    )


class RAGExpertInputWrapper(BaseModel):
    text_chunks: StrictStr
    query_text: StrictStr
    output_language: StrictStr


class RAGExpertOutput(BaseModel):
    used_chunk_ids: list[PositiveInt] = []
    response: Optional[StrictStr] = None
