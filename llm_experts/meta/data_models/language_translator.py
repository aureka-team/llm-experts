from iso639 import Language, LanguageNotFoundError
from pydantic import BaseModel, StrictStr, field_validator


def languge_validator(language: str) -> str:
    try:
        return Language.from_name(language).name
    except LanguageNotFoundError:
        raise ValueError(f"{language} isn't an ISO language name")


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
