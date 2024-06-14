from typing import Optional

from common.cache import RedisCache
from common.logger import get_logger

from llm_experts.data import experts
from llm_experts.meta import (
    OpenAIChatExpert,
    LanguageTranslatorInput,
    LanguageTranslatorOutput,
)


logger = get_logger(__name__)


class LanguageTranslatorExpert(OpenAIChatExpert):
    def __init__(
        self,
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path=f"{experts.__path__[0]}/language-translator.yaml",
            expert_output=LanguageTranslatorOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: LanguageTranslatorInput,
    ) -> LanguageTranslatorOutput:
        return self._generate(expert_input=expert_input)
