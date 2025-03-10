from typing import Literal
from pydantic import BaseModel, StrictStr

from common.cache import RedisCache
from common.logger import get_logger

from llm_experts.conf import experts
from llm_experts.meta.interfaces import LLMExpert


logger = get_logger(__name__)


class ImageDescriberInput(BaseModel):
    image_url: StrictStr
    image_detail: Literal["low", "high"]


class ImageDescriberOutput(BaseModel):
    description: StrictStr


class ImageDescriber(LLMExpert):
    def __init__(
        self,
        conf_path=f"{experts.__path__[0]}/image-describer.yaml",
        max_concurrency: int = 10,
        cache: RedisCache = None,
    ):
        super().__init__(
            conf_path=conf_path,
            expert_output=ImageDescriberOutput,
            max_concurrency=max_concurrency,
            cache=cache,
        )

    def generate(
        self,
        expert_input: ImageDescriberInput,
    ) -> ImageDescriberOutput:
        return self._generate(expert_input=expert_input)
