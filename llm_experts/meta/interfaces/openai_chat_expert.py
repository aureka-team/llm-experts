import asyncio
import tiktoken

from tqdm import tqdm
from joblib import hash

from typing import Optional
from pydantic import BaseModel
from redis_cache import RedisCache


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


from llm_experts.logger import get_logger
from llm_experts.utils.yaml_data import load_yaml
from llm_experts.exceptions import LLMResponseError


logger = get_logger(__name__)


class OpenAIChatExpert:
    def __init__(
        self,
        conf_path: str,
        expert_output: BaseModel,
        max_concurrency: int = 10,
        retry_conf_path: str = "/resources/llm-experts/expert-output-parser.yaml",  # noqa
        cache: Optional[RedisCache] = None,
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
            pydantic_object=expert_output
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

    def _get_token_len(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _get_cache_key(self, expert_input: BaseModel) -> str:
        return hash(f"{hash(self.conf)}-{hash(expert_input)}")

    def generate(self, expert_input: BaseModel) -> BaseModel:
        cache_key = self._get_cache_key(expert_input=expert_input)
        if self.cache is not None:
            response_text = self.cache.load(cache_key)
            if response_text is not None:
                return response_text

        logger.debug(f"waiting for {self.name} response")
        prompt = self.chat_prompt.format_prompt(**expert_input.model_dump())
        messages = prompt.to_messages()

        system_prompt = messages[0].content
        human_prompt = messages[1].content

        logger.debug(f"system_prompt => {system_prompt}")
        logger.debug(f"human_prompt => {human_prompt}")

        prompt_token_len = self._get_token_len(
            system_prompt
        ) + self._get_token_len(human_prompt)

        logger.debug(f"prompt_token_len => {prompt_token_len}")
        with get_openai_callback() as callback:
            llm_response = self.llm.invoke(messages)
            parsed_callback = {
                "completion_tokens": callback.completion_tokens,
                "prompt_tokens": callback.prompt_tokens,
                "total_tokens": callback.total_tokens,
                "total_cost": callback.total_cost,
            }

            logger.info(f"openai_callback => {parsed_callback}")

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
            self.cache.save(obj=pydantic_output, cache_key=cache_key)

        return pydantic_output

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
