from pydantic import BaseModel, StrictStr, PositiveFloat
from pydantic_extra_types.language_code import LanguageAlpha2

from common.cache import RedisCache
from common.logger import get_logger

from llm_experts.conf import experts
from llm_experts.meta.interfaces import LLMExpert


logger = get_logger(__name__)


class LanguageDetectorInput(BaseModel):
    text: StrictStr


class LanguageDetectorOutput(BaseModel):
    language: LanguageAlpha2 | None = None
    confidence: PositiveFloat = None


class LanguageDetectorExpert(LLMExpert):
    def __init__(
        self,
        conf_path=f"{experts.__path__[0]}/language-detector.yaml",
        max_concurrency: int = 10,
        cache: RedisCache | None = None,
    ):
        super().__init__(
            conf_path=conf_path,
            expert_output=LanguageDetectorOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: LanguageDetectorInput,
    ) -> LanguageDetectorOutput:
        return self._generate(expert_input=expert_input)
