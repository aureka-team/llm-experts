from pydantic import BaseModel, StrictStr, field_validator

from .validators import languge_validator


class LanguageTranslatorExpertInput(BaseModel):
    source_text: StrictStr
    source_language: StrictStr
    target_language: StrictStr

    _source_language_validator = field_validator("source_language")(
        languge_validator
    )

    _target_language_validator = field_validator("target_language")(
        languge_validator
    )


class LanguageTranslatorExpertOutput(BaseModel):
    translation: StrictStr
