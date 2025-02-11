from pydantic import BaseModel, StrictStr

from common.logger import get_logger
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from llm_experts.conf import experts
from llm_experts.meta.interfaces import LLMExpert


logger = get_logger(__name__)


class GenericAssistantInput(BaseModel):
    user_query: StrictStr


class GenericAssistantOutput(BaseModel):
    assistant_response: StrictStr


class GenericAssistant(LLMExpert):
    def __init__(
        self,
        conf_path: str = f"{experts.__path__[0]}/generic-assistant.yaml",
        max_concurrency: int = 10,
        mongodb_chat_history: MongoDBChatMessageHistory | None = None,
        read_only_history: bool = False,
    ):
        super().__init__(
            conf_path=conf_path,
            expert_output=GenericAssistantOutput,
            input_messages_key="user_query",
            max_concurrency=max_concurrency,
            mongodb_chat_history=mongodb_chat_history,
            read_only_history=read_only_history,
        )

    def generate(
        self,
        expert_input: GenericAssistantInput,
    ) -> GenericAssistantOutput:
        return self._generate(expert_input=expert_input)
