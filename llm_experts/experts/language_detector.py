from typing import Optional

from redis_cache import RedisCache
from llm_experts.logger import get_logger
from llm_experts.meta import OpenAIChatExpert, LanguageDetectorExpertOutput


logger = get_logger(__name__)


class LanguageDetectorExpert(OpenAIChatExpert):
    def __init__(
        self,
        conf_path: str = "/resources/llm-experts/language-detector.yaml",
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path=conf_path,
            expert_output=LanguageDetectorExpertOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )
