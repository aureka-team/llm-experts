import asyncio
import tiktoken

from joblib import hash

from rich.pretty import pprint
from typing import Any, Optional
from abc import ABC, abstractmethod

from langchain_openai.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import OutputParserException
from langchain.output_parsers import (
    PydanticOutputParser,
    RetryWithErrorOutputParser,
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from llm_experts.cache import Cache
from llm_experts.logger import get_logger
from llm_experts.utils.yaml_data import load_yaml
from llm_experts.exceptions import LLMResponseError
from llm_experts.utils.langchain import parse_openai_callback


logger = get_logger(__name__)


class OpenAIChatLLM(ABC):
    def __init__(
        self,
        conf_path: str,
        pydantic_output_object: Any,
        max_concurrency: int = 5,
        retry_conf_path: str = "/resources/llm/experts/output-parser.yaml",
        cache: Optional[Cache] = None,
    ):
        self.conf = load_yaml(conf_path)
        self.cache = cache
        self.semaphore = asyncio.Semaphore(max_concurrency)

        model_name = self.conf["model-name"]
        self.llm = ChatOpenAI(
            model_name=model_name,
            max_tokens=self.conf["max-tokens"],
            temperature=self.conf["temperature"],
        )

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.conf["system-prompt-template"]
        )

        human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.conf["human-prompt-template"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        encoder_name = self.conf.get("encoder-name", model_name)
        self.enc = tiktoken.encoding_for_model(encoder_name)
        self.output_parser = PydanticOutputParser(
            pydantic_object=pydantic_output_object
        )

        self.retry_output_parser = RetryWithErrorOutputParser.from_llm(
            parser=self.output_parser,
            llm=self.llm,
            prompt=PromptTemplate.from_template(
                template=load_yaml(retry_conf_path)["base-prompt"]
            ),
        )

    @property
    def name(self):
        return self.__class__.__name__

    def get_token_len(self, text: str) -> int:
        return len(self.enc.encode(text))

    def get_cache_key(self, chat_prompt_params: dict) -> str:
        conf_part = hash(self.conf)
        prompt_part = hash(chat_prompt_params)
        cache_key = hash(f"{conf_part}-{prompt_part}")
        return cache_key

    def get_llm_response(self, chat_prompt_params: dict) -> Any:
        cache_key = self.get_cache_key(chat_prompt_params)
        if self.cache is not None:
            response_text = self.cache.load(cache_key)
            if response_text is not None:
                return response_text

        logger.debug(f"waiting for {self.name} response")

        prompt = self.chat_prompt.format_prompt(**chat_prompt_params)
        messages = prompt.to_messages()

        system_prompt = messages[0].content
        human_prompt = messages[1].content

        logger.debug(f"system_prompt => {system_prompt}")
        logger.debug(f"human_prompt => {human_prompt}")

        prompt_token_len = self.get_token_len(
            system_prompt
        ) + self.get_token_len(human_prompt)

        logger.debug(f"prompt_token_len => {prompt_token_len}")
        with get_openai_callback() as callback:
            llm_response = self.llm.invoke(messages)

            openai_callback = parse_openai_callback(callback)
            pprint(openai_callback)

        response_text = llm_response.content
        logger.debug(f"response_text => {response_text}")

        try:
            pydantic_output = self.output_parser.parse(text=response_text)
        except OutputParserException:
            logger.warning(
                f"trying to fix malformed llm response => {response_text}"
            )

            try:
                pydantic_output = self.retry_output_parser.parse_with_prompt(
                    response_text, prompt
                )

            except OutputParserException:
                raise LLMResponseError(response_text=response_text)

        if self.cache is not None:
            self.cache.save(cache_key, pydantic_output)

        return pydantic_output

    @abstractmethod
    def generate(self) -> Any:
        pass

    async def async_generate(self, *args, **kwargs) -> Any:
        return await asyncio.to_thread(self.generate, *args, **kwargs)
