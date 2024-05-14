from typing import Optional

from redis_cache import RedisCache
from llm_experts.logger import get_logger
from llm_experts.utils.json_data import get_pretty
from llm_experts.meta import (
    OpenAIChatExpert,
    RAGExpertInput,
    RAGExpertInputWrapper,
    RAGExpertOutput,
)


logger = get_logger(__name__)


class RAGExpert(OpenAIChatExpert):
    def __init__(
        self,
        conf_path: str = "/resources/llm-experts/rag.yaml",
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path=conf_path,
            expert_output=RAGExpertOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def expert_input_wrapper(
        self,
        expert_input: RAGExpertInput,
    ) -> RAGExpertInputWrapper:
        expert_input_dict = expert_input.model_dump()
        return RAGExpertInputWrapper(
            text_chunks=get_pretty(obj=expert_input_dict["text_chunks"]),
            query_text=expert_input_dict["query_text"],
            output_language=expert_input_dict["output_language"],
        )

    def generate(
        self,
        expert_input: RAGExpertInput,
    ) -> RAGExpertOutput:
        return self._generate(
            self.expert_input_wrapper(expert_input=expert_input)
        )
