import asyncio

from tqdm import tqdm
from joblib import hash

from typing import Literal
from abc import ABC, abstractmethod
from pydantic import (
    BaseModel,
    StrictStr,
    PositiveInt,
    NonNegativeFloat,
    Field,
    ConfigDict,
)

from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.callbacks import get_openai_callback
from langchain.schema.output_parser import OutputParserException
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
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
from llm_experts.exceptions import (
    LLMResponseError,
    ParameterDependencyException,
)


logger = get_logger(__name__)


MODEL_TYPE_MAP = {
    "gpt": ChatOpenAI,
    "google": ChatGoogleGenerativeAI,
    "llama": ChatOllama,
}


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: Literal["gpt", "google", "llama"] = Field(alias="model-type")
    model: StrictStr = Field(alias="model-name")
    temperature: NonNegativeFloat
    max_tokens: PositiveInt | None = Field(
        alias="max-tokens",
        default=None,
    )

    system_prompt_template: StrictStr | None = Field(
        alias="system-prompt-template",
        default=None,
    )

    human_prompt_template: StrictStr | None = Field(
        alias="human-prompt-template",
        default=None,
    )

    base_prompt: StrictStr | None = Field(
        alias="base-prompt",
        default=None,
    )


# TODO: Add limit by session_id to the mongodb_chat_history
class LLMExpert(ABC):
    def __init__(
        self,
        conf_path: str,
        expert_output: BaseModel,
        input_messages_key: str | None = None,
        max_concurrency: int = 10,
        retry_conf_path: str = f"{experts.__path__[0]}/output-parser.yaml",
        cache: RedisCache | None = None,
        mongodb_chat_history: MongoDBChatMessageHistory | None = None,
        read_only_history: bool = False,
    ):
        self.input_messages_key = input_messages_key
        self.cache = cache
        self.mongodb_chat_history = mongodb_chat_history
        self.read_only_history = read_only_history

        self.with_message_history = (
            True if mongodb_chat_history is not None else False
        )

        if self.with_message_history and self.input_messages_key is None:
            raise ParameterDependencyException(
                dependent_param="mongodb_chat_history",
                required_param="input_messages_key",
            )

        # TODO: Is this ok?
        if self.with_message_history and self.cache is not None:
            self.cache = None
            logger.warning(
                (
                    "cache will be ignored when `mongodb_chat_history` "
                    "is provided"
                )
            )

        self.conf = Config(**load_yaml(file_path=conf_path))
        self.prompt_messages = [
            SystemMessagePromptTemplate.from_template(
                self.conf.system_prompt_template,
            ),
            HumanMessagePromptTemplate.from_template(
                self.conf.human_prompt_template
            ),
        ]

        if self.with_message_history:
            self.prompt_messages.insert(
                -1,
                MessagesPlaceholder(variable_name="history"),
            )

        self.llm = self._get_chat_llm(conf=self.conf)
        self.output_parser = PydanticOutputParser(pydantic_object=expert_output)
        output_parser_conf = Config(
            **(
                load_yaml(file_path=retry_conf_path)
                | {"max-tokens": self.conf.max_tokens}
            )
        )

        logger.info(output_parser_conf)
        output_parser_llm = self._get_chat_llm(conf=output_parser_conf)
        self.retry_output_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.output_parser,
            llm=output_parser_llm,
            prompt=PromptTemplate.from_template(
                template=output_parser_conf.base_prompt
            ),
        )

        self.semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # FIXME: This method should be improved.
    def _get_chat_llm(
        self,
        conf: Config,
    ) -> ChatOpenAI | ChatGoogleGenerativeAI | ChatOllama:
        dict_conf = conf.model_dump()
        dict_conf.pop("base_prompt")
        dict_conf.pop("system_prompt_template")
        dict_conf.pop("human_prompt_template")

        model_type = dict_conf.pop("model_type")
        if model_type == "llama":
            dict_conf["num_predict"] = dict_conf.pop("max_tokens")

        return MODEL_TYPE_MAP[model_type](**dict_conf)

    def _get_cache_key(self, expert_input: BaseModel) -> str:
        return hash(f"{hash(self.conf)}-{hash(expert_input)}")

    # TODO: There is a better way to do this?
    def _remove_last_history_message(self) -> None:
        last_message_rows = self.mongodb_chat_history.collection.find(
            {"SessionId": self.mongodb_chat_history.session_id},
            sort={"_id": -1},
            limit=2,
        )

        for row in last_message_rows:
            self.mongodb_chat_history.collection.delete_one({"_id": row["_id"]})

    def _generate(self, expert_input: BaseModel) -> BaseModel:
        cache_key = self._get_cache_key(expert_input=expert_input)
        if self.cache is not None:
            response_text = self.cache.load(cache_key)
            if response_text is not None:
                return response_text

        logger.debug(f"waiting for {self.name} response")
        chat_prompt = ChatPromptTemplate.from_messages(self.prompt_messages)
        runnable = chat_prompt | self.llm
        if self.mongodb_chat_history is not None:
            runnable = RunnableWithMessageHistory(
                runnable,
                lambda session_id: self.mongodb_chat_history,
                input_messages_key=self.input_messages_key,
                history_messages_key="history",
            )

        with get_openai_callback() as callback:
            config = (
                {
                    "configurable": {
                        "session_id": self.mongodb_chat_history.session_id,
                    }
                }
                if self.with_message_history
                else None
            )

            llm_response = runnable.invoke(
                input=expert_input.model_dump(),
                config=config,
            )

            parsed_callback = {
                "completion_tokens": callback.completion_tokens,
                "prompt_tokens": callback.prompt_tokens,
                "total_tokens": callback.total_tokens,
                "total_cost": callback.total_cost,
            }

            logger.debug(f"openai_callback => {parsed_callback}")

        if self.read_only_history:
            self._remove_last_history_message()

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

        if self.cache is not None:
            self.cache.save(obj=pydantic_output, cache_key=cache_key)

        return pydantic_output

    @abstractmethod
    def generate(self, expert_input: BaseModel) -> BaseModel:
        pass

    async def async_generate(
        self,
        expert_input: BaseModel,
        pbar: tqdm | None = None,
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
