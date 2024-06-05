from typing import Optional

from common.cache import RedisCache
from common.logger import get_logger

from llm_experts.meta import (
    OpenAIChatExpert,
    RAGInput,
    RAGOutput,
)


logger = get_logger(__name__)


class LupAIAssistant(OpenAIChatExpert):
    def __init__(
        self,
        with_message_history=True,
        input_messages_key="query_text",
        max_messages=20,
        max_concurrency: int = 10,
        cache: Optional[RedisCache] = None,
    ):
        super().__init__(
            conf_path="experts/lupai-assistant.yaml",
            expert_output=RAGOutput,
            with_message_history=with_message_history,
            input_messages_key=input_messages_key,
            max_messages=max_messages,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: RAGInput,
    ) -> RAGOutput:
        return self._generate(expert_input=expert_input)
