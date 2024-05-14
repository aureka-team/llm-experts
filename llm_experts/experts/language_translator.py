from typing import Optional

from redis_cache import RedisCache
from llm_experts.logger import get_logger
from llm_experts.meta import (
    OpenAIChatExpert,
    LanguageTranslatorExpertInput,
    LanguageTranslatorExpertOutput,
)


logger = get_logger(__name__)


class LanguageTranslatorExpert(OpenAIChatExpert):
    def __init__(
        self,
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path="experts/language-translator.yaml",
            expert_output=LanguageTranslatorExpertOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: LanguageTranslatorExpertInput,
    ) -> LanguageTranslatorExpertOutput:
        return self._generate(expert_input=expert_input)
