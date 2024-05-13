import asyncio

import numpy as np

from tqdm import tqdm

from joblib import hash
from typing import Optional

from abc import ABC, abstractmethod
from more_itertools import chunked, flatten

from llm_experts.logger import get_logger
from llm_experts.utils.path import create_path
from llm_experts.utils.h5_files import save_h5, load_h5


logger = get_logger(__name__)


class TextEncoder(ABC):
    def __init__(
        self,
        batch_size: int = 1024,
        max_concurrency: int = 10,
        cache_path: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.cache_path = cache_path

        if cache_path is not None:
            create_path(path=cache_path)

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray:
        pass

    def _get_cache_key(self, texts: list[str]) -> str:
        model_name_hash = hash(self.model_name)
        texts_hash = hash(texts)
        cache_key = hash(f"{model_name_hash}-{texts_hash}")

        return cache_key

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.cache_path is not None:
            cache_key = self._get_cache_key(texts=texts)
            loaded_vectors = load_h5(self.cache_path, cache_key)
            if loaded_vectors is not None:
                return loaded_vectors

        vectors = self._encode(texts=texts)
        if self.cache_path is not None:
            save_h5(self.cache_path, cache_key, vectors)

        return vectors

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        logger.info("generating text vectors...")
        text_chunks = chunked(texts, self.batch_size)
        chunk_vectors = map(
            self.encode,
            tqdm(
                text_chunks,
                total=(len(texts) // self.batch_size),
                ascii=" ##",
                colour="#808080",
            ),
        )

        text_vetors = np.array(list(flatten(chunk_vectors)))
        return text_vetors

    async def async_encode(
        self,
        texts: list[str],
        pbar: tqdm,
    ) -> np.ndarray:
        async with self.semaphore:
            vectors = await asyncio.to_thread(self.encode, texts)

            pbar.update(1)
            return vectors

    async def async_batch_encode(self, texts: list[str]) -> np.ndarray:
        logger.info("generating text vectors...")
        text_chunks = chunked(texts, self.batch_size)
        with tqdm(
            text_chunks,
            total=(len(texts) // self.batch_size),
            ascii=" ##",
            colour="#808080",
        ) as pbar:

            async_tasks = [
                self.async_encode(texts, pbar=pbar) for texts in text_chunks
            ]

            chunk_vectors = await asyncio.gather(*async_tasks)

        text_vetors = np.array(list(flatten(chunk_vectors)))
        return text_vetors
