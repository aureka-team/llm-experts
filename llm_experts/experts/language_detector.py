from typing import Optional

from common.cache import RedisCache
from common.logger import get_logger

from llm_experts.conf import experts
from llm_experts.meta import (
    OpenAIChatExpert,
    LanguageDetectorInput,
    LanguageDetectorOutput,
)


logger = get_logger(__name__)


class LanguageDetectorExpert(OpenAIChatExpert):
    def __init__(
        self,
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path=f"{experts.__path__[0]}/language-detector.yaml",
            expert_output=LanguageDetectorOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: LanguageDetectorInput,
    ) -> LanguageDetectorOutput:
        return self._generate(expert_input=expert_input)
