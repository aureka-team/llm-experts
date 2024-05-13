from pydantic import BaseModel, StrictStr


class LanguageTranslatorExpertOutput(BaseModel):
    translation: StrictStr
