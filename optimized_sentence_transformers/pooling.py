import json
from pathlib import Path
from typing import Optional

import numpy as np


class Pooling:
    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode_cls_token: bool,
        pooling_mode_mean_tokens: bool,
        pooling_mode_max_tokens: bool,
        pooling_mode_mean_sqrt_len_tokens: bool,
    ) -> None:
        self._word_embedding_dimension = word_embedding_dimension
        self._pooling_mode_cls_token = pooling_mode_cls_token
        self._pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self._pooling_mode_max_tokens = pooling_mode_max_tokens
        self._pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

    def __call__(
        self,
        attention_mask: np.ndarray,
        token_embeddings: np.ndarray,
        cls_token_embeddings: Optional[np.ndarray] = None,
        token_weights_sum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        output_vectors = []

        if self._pooling_mode_cls_token:
            output_vectors.append(
                self._cls_token_pooling(
                    token_embeddings,
                    cls_token_embeddings,
                ),
            )

        if self._pooling_mode_max_tokens:
            output_vectors.append(
                self._max_token_pooling(
                    attention_mask,
                    token_embeddings,
                ),
            )

        if self._pooling_mode_mean_tokens or self._pooling_mode_mean_sqrt_len_tokens:
            output_vectors.append(
                self._mean_pooling(
                    attention_mask,
                    token_embeddings,
                    token_weights_sum,
                ),
            )

        return np.concatenate(
            output_vectors,
            axis=1,
        )

    @classmethod
    def from_config(
        cls,
        config_path: Path,
    ) -> 'Pooling':
        with open(config_path, 'r', encoding='utf-8') as inp:
            config = json.load(inp)

        return cls(
            word_embedding_dimension=config['word_embedding_dimension'],
            pooling_mode_cls_token=config['pooling_mode_cls_token'],
            pooling_mode_mean_tokens=config['pooling_mode_mean_tokens'],
            pooling_mode_max_tokens=config['pooling_mode_max_tokens'],
            pooling_mode_mean_sqrt_len_tokens=config['pooling_mode_mean_sqrt_len_tokens'],
        )

    @staticmethod
    def _cls_token_pooling(
        token_embeddings: np.ndarray,
        cls_token_embeddings: Optional[np.ndarray],
    ) -> np.ndarray:
        if cls_token_embeddings is None:
            return token_embeddings[:, 0]

        return cls_token_embeddings

    def _max_token_pooling(
        self,
        attention_mask: np.ndarray,
        token_embeddings: np.ndarray,
    ) -> np.ndarray:
        expanded_mask = self._expand(
            attention_mask,
            token_embeddings.shape,
        )
        token_embeddings[expanded_mask == 0] = -1e-9

        return np.max(
            token_embeddings,
            axis=1,
        )[0]

    def _mean_pooling(  # type: ignore
        self,
        attention_mask: np.ndarray,
        token_embeddings: np.ndarray,
        token_weights_sum: Optional[np.ndarray],
    ) -> np.ndarray:
        expanded_mask = self._expand(
            attention_mask,
            token_embeddings.shape,
        )
        embeddings_sum = np.sum(
            token_embeddings,
            axis=1,
        )

        if token_weights_sum is None:
            mask_sum = np.sum(
                expanded_mask,
                axis=1,
            )
        else:
            mask_sum = self._expand(
                token_weights_sum,
                embeddings_sum.shape,
            )

        mask_sum = np.maximum(
            mask_sum,
            1e-9,  # noqa: WPS432
        )

        if self._pooling_mode_mean_tokens:
            return embeddings_sum / mask_sum

        if self._pooling_mode_mean_sqrt_len_tokens:
            return embeddings_sum / np.sqrt(mask_sum)

    @staticmethod
    def _expand(
        arr: np.ndarray,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        return np.broadcast_to(
            arr[..., np.newaxis],
            shape,
        ).astype(float)
