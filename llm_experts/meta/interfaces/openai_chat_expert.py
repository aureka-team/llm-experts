import uuid
import asyncio

from tqdm import tqdm
from joblib import hash

from typing import Optional
from pydantic import BaseModel
from more_itertools import chunked
from abc import ABC, abstractmethod

from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.callbacks import get_openai_callback
from langchain.schema.output_parser import OutputParserException
from langchain.output_parsers import (
    PydanticOutputParser,
    RetryWithErrorOutputParser,
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


from common.cache import RedisCache
from common.logger import get_logger
from common.utils.yaml_data import load_yaml

from llm_experts.conf import experts
from llm_experts.store import MongoStore
from llm_experts.meta import StoreMessages
from llm_experts.exceptions import (
    LLMResponseError,
    ParameterDependencyException,
)


logger = get_logger(__name__)


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    session_id: str
    max_messages: int
    messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self.messages = self.messages[-self.max_messages :]  # noqa

    def clear(self) -> None:
        self.messages = []


class OpenAIChatExpert(ABC):
    def __init__(
        self,
        conf_path: str,
        expert_output: BaseModel,
        with_message_history: bool = False,
        input_messages_key: Optional[str] = None,
        max_messages: int = 20,
        max_concurrency: int = 10,
        retry_conf_path: str = f"{experts.__path__[0]}/output-parser.yaml",  # noqa
        cache: Optional[RedisCache] = None,
        mongo_store: Optional[MongoStore] = None,
    ):
        self.with_message_history = with_message_history
        self.input_messages_key = input_messages_key
        self.max_messages = max_messages
        if self.with_message_history and self.input_messages_key is None:
            raise ParameterDependencyException(
                dependent_param="with_message_history",
                required_param="input_messages_key",
            )

        self.cache = cache
        self.mongo_store = mongo_store

        if self.with_message_history and self.cache is not None:
            self.cache = None
            logger.warning(
                (
                    "cache will be ignored when `with_message_history` "
                    "is enabled"
                )
            )

        self.conf = load_yaml(file_path=conf_path)
        self.semaphore = asyncio.Semaphore(max_concurrency)

        self.prompt_messages = [
            SystemMessagePromptTemplate.from_template(
                self.conf["system-prompt-template"],
            ),
            HumanMessagePromptTemplate.from_template(
                self.conf["human-prompt-template"]
            ),
        ]

        if self.with_message_history:
            self.prompt_messages.insert(
                -1,
                MessagesPlaceholder(variable_name="history"),
            )

        self.llm = ChatOpenAI(
            model_name=self.conf["model-name"],
            max_tokens=self.conf["max-tokens"],
            temperature=self.conf["temperature"],
        )

        self.output_parser = PydanticOutputParser(
            pydantic_object=expert_output
        )

        output_parser_conf = load_yaml(file_path=retry_conf_path)
        self.retry_output_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.output_parser,
            llm=ChatOpenAI(
                model_name=output_parser_conf["model-name"],
                max_tokens=self.conf["max-tokens"],
                temperature=output_parser_conf["temperature"],
            ),
            prompt=PromptTemplate.from_template(
                template=output_parser_conf["base-prompt"]
            ),
        )

        self.history = InMemoryHistory(
            session_id=uuid.uuid4().hex,
            max_messages=self.max_messages,
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _get_cache_key(self, expert_input: BaseModel) -> str:
        return hash(f"{hash(self.conf)}-{hash(expert_input)}")

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        return self.history

    def _parse_message_pair(self, message_pair: tuple[BaseMessage]) -> dict:
        human_message = message_pair[0]
        ai_message = message_pair[1]
        token_usage = ai_message.response_metadata["token_usage"]

        return {
            "human_message": human_message.content,
            "ai_message": ai_message.content,
            "response_metadata": {
                "model": ai_message.response_metadata["model_name"],
                "completion_tokens": token_usage["completion_tokens"],
                "prompt_tokens": token_usage["prompt_tokens"],
                "total_tokens": token_usage["total_tokens"],
            },
        }

    def _save_messages(self) -> None:
        message_pairs = chunked(iterable=self.history.messages, n=2)
        self.mongo_store.save_message_history(
            store_messages=StoreMessages(
                session_id=self.history.session_id,
                messages=[
                    self._parse_message_pair(message_pair=message_pair)
                    for message_pair in message_pairs
                ],
            )
        )

    def _generate(self, expert_input: BaseModel) -> BaseModel:
        cache_key = self._get_cache_key(expert_input=expert_input)
        if self.cache is not None:
            response_text = self.cache.load(cache_key)
            if response_text is not None:
                return response_text

        logger.debug(f"waiting for {self.name} response")
        chat_prompt = ChatPromptTemplate.from_messages(self.prompt_messages)
        runnable = chat_prompt | self.llm
        if self.with_message_history:
            runnable = RunnableWithMessageHistory(
                runnable,
                self._get_session_history,
                input_messages_key=self.input_messages_key,
                history_messages_key="history",
            )

        with get_openai_callback() as callback:
            llm_response = runnable.invoke(
                input=expert_input.model_dump(),
                config={
                    "configurable": {
                        "session_id": self.history.session_id,
                    },
                },
            )

            parsed_callback = {
                "completion_tokens": callback.completion_tokens,
                "prompt_tokens": callback.prompt_tokens,
                "total_tokens": callback.total_tokens,
                "total_cost": callback.total_cost,
            }

            logger.info(f"openai_callback => {parsed_callback}")

        response_text = llm_response.content

        try:
            pydantic_output = self.output_parser.parse(text=response_text)

        except OutputParserException:
            logger.warning(
                f"trying to fix malformed llm response => {response_text}"
            )

            try:
                pydantic_output = self.retry_output_parser.parse_with_prompt(
                    response_text,
                    chat_prompt.format_prompt(**expert_input.model_dump()),
                )

            except OutputParserException:
                raise LLMResponseError(response_text=response_text)

        logger.debug(f"history messages => {len(self.history.messages)}")
        if self.cache is not None:
            self.cache.save(obj=pydantic_output, cache_key=cache_key)

        if self.mongo_store is not None:
            self._save_messages()

        return pydantic_output

    @abstractmethod
    def generate(self, expert_input: BaseModel) -> BaseModel:
        pass

    async def async_generate(
        self,
        expert_input: BaseModel,
        pbar: Optional[tqdm] = None,
    ) -> BaseModel:
        async with self.semaphore:
            expert_output = await asyncio.to_thread(
                self.generate, expert_input=expert_input
            )

            if pbar is not None:
                pbar.update(1)

            return expert_output

    async def batch_generate(
        self,
        expert_inputs: list[BaseModel],
    ) -> list[BaseModel]:
        with tqdm(
            total=len(expert_inputs),
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            pass

            async_tasks = [
                self.async_generate(
                    expert_input=expert_input,
                    pbar=pbar,
                )
                for expert_input in expert_inputs
            ]

            return await asyncio.gather(*async_tasks)
