from typing import Optional

from redis_cache import RedisCache
from llm_experts.logger import get_logger
from llm_experts.meta import (
    OpenAIChatExpert,
    LanguageDetectorInput,
    LanguageDetectorExpertOutput,
)


logger = get_logger(__name__)


class LanguageDetectorExpert(OpenAIChatExpert):
    def __init__(
        self,
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path="experts/language-detector.yaml",
            expert_output=LanguageDetectorExpertOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: LanguageDetectorInput,
    ) -> LanguageDetectorExpertOutput:
        return self._generate(expert_input=expert_input)
