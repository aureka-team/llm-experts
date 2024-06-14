from typing import Optional

from common.cache import RedisCache
from common.logger import get_logger
from common.utils.json_data import get_pretty

from llm_experts.data import experts
from llm_experts.meta import (
    OpenAIChatExpert,
    RAGInput,
    RAGInputWrapper,
    RAGOutput,
)


logger = get_logger(__name__)


class RAGExpert(OpenAIChatExpert):
    def __init__(
        self,
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path=f"{experts.__path__[0]}/rag.yaml",
            expert_output=RAGOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def expert_input_wrapper(
        self,
        expert_input: RAGInput,
    ) -> RAGInputWrapper:
        expert_input_dict = expert_input.model_dump()
        return RAGInputWrapper(
            text_chunks=get_pretty(obj=expert_input_dict["text_chunks"]),
            query_text=expert_input_dict["query_text"],
            output_language=expert_input_dict["output_language"],
        )

    def generate(
        self,
        expert_input: RAGInput,
    ) -> RAGOutput:
        return self._generate(
            self.expert_input_wrapper(expert_input=expert_input)
        )
