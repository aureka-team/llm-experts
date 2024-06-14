import uuid
import asyncio

from tqdm import tqdm
from joblib import hash

from typing import Optional
from pydantic import BaseModel
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

from llm_experts.data import experts
from llm_experts.exceptions import (
    LLMResponseError,
    ParameterDependencyException,
)


logger = get_logger(__name__)


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
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
        retry_conf_path: str = f"{experts.__path__[0]}/expert-output-parser.yaml",  # noqa
        cache: Optional[RedisCache] = None,
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

        self.retry_output_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.output_parser,
            llm=self.llm,
            prompt=PromptTemplate.from_template(
                template=load_yaml(file_path=retry_conf_path)["base-prompt"]
            ),
        )

        self.message_history_store = {}
        self.session_id = uuid.uuid4().hex

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _get_cache_key(self, expert_input: BaseModel) -> str:
        return hash(f"{hash(self.conf)}-{hash(expert_input)}")

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.message_history_store:
            self.message_history_store[session_id] = InMemoryHistory(
                max_messages=self.max_messages
            )

        return self.message_history_store[session_id]

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
                        "session_id": self.session_id,
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

        history_messages = self.message_history_store.get(self.session_id)
        n_history_messages = (
            0 if history_messages is None else len(history_messages.messages)
        )

        logger.debug(f"history messages => {n_history_messages}")

        if self.cache is not None:
            self.cache.save(obj=pydantic_output, cache_key=cache_key)

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
