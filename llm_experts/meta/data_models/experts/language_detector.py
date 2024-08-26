from typing import Optional
from pycountry import languages

from pydantic import BaseModel, StrictStr, PositiveFloat, field_validator


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageDetectorOutput(BaseModel):
    language: Optional[StrictStr] = None
    confidence: Optional[PositiveFloat] = None

    @field_validator("language", mode="before")
    def language_validator(cls, v: str) -> str | None:
        print(v)
        if v is not None:
            language_code = languages.get(alpha_2=v)
            if language_code is not None:
                return language_code.alpha_2
