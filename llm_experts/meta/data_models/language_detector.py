from typing import Optional
from iso639 import Language, LanguageNotFoundError

from pydantic import BaseModel, StrictStr, PositiveFloat, field_validator


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageDetectorExpertOutput(BaseModel):
    language: Optional[StrictStr] = None
    confidence: Optional[PositiveFloat] = None

    @field_validator("language", mode="before")
    def language_validator(cls, v: str) -> str:
        if v is not None:
            try:
                _ = Language.from_part1(v)
            except LanguageNotFoundError:
                return "en"

        return v
