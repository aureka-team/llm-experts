from pydantic import BaseModel, StrictStr, PositiveFloat
from pydantic_extra_types.language_code import LanguageAlpha2


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageDetectorOutput(BaseModel):
    language: LanguageAlpha2 | None = None
    confidence: PositiveFloat = None
