from pydantic import BaseModel, StrictStr, field_validator

from llm_experts.utils.validators import languge_validator


class LanguageTranslatorInput(BaseModel):
    source_text: StrictStr
    source_language: StrictStr
    target_language: StrictStr

    _source_language_validator = field_validator("source_language")(
        languge_validator
    )

    _target_language_validator = field_validator("target_language")(
        languge_validator
    )


class LanguageTranslatorOutput(BaseModel):
    translation: StrictStr
