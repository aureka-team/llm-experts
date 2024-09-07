from pydantic import BaseModel, StrictStr
from pydantic_extra_types.language_code import LanguageName


class LanguageTranslatorInput(BaseModel):
    source_text: StrictStr
    source_language: LanguageName
    target_language: LanguageName


class LanguageTranslatorOutput(BaseModel):
    translation: StrictStr
