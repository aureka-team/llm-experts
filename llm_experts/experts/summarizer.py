from typing import Optional

from common.cache import RedisCache
from common.logger import get_logger

from llm_experts.meta import (
    OpenAIChatExpert,
    SummarizerExpertInput,
    SummarizerExpertOutput,
)


logger = get_logger(__name__)


class SummarizerExpert(OpenAIChatExpert):
    def __init__(
        self,
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path="experts/summarizer.yaml",
            expert_output=SummarizerExpertOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: SummarizerExpertInput,
    ) -> SummarizerExpertOutput:
        return self._generate(expert_input=expert_input)
