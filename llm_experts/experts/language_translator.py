from pydantic import BaseModel, StrictStr
from pydantic_extra_types.language_code import LanguageName

from common.cache import RedisCache
from common.logger import get_logger

from llm_experts.conf import experts
from llm_experts.meta.interfaces import LLMExpert


logger = get_logger(__name__)


class LanguageTranslatorInput(BaseModel):
    source_text: StrictStr
    source_language: LanguageName
    target_language: LanguageName


class LanguageTranslatorOutput(BaseModel):
    translation: StrictStr


class LanguageTranslatorExpert(LLMExpert):
    def __init__(
        self,
        conf_path=f"{experts.__path__[0]}/language-translator.yaml",
        max_concurrency: int = 10,
        cache: RedisCache = None,
    ):
        super().__init__(
            conf_path=conf_path,
            expert_output=LanguageTranslatorOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: LanguageTranslatorInput,
    ) -> LanguageTranslatorOutput:
        return self._generate(expert_input=expert_input)
