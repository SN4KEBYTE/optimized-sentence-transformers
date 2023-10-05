import os
import warnings
from pathlib import Path
from typing import (
    Any,
    Literal,
    Optional,
    Union,
)

import numpy as np
import onnxruntime as ort
import psutil
from huggingface_hub import snapshot_download
from tqdm import trange
from transformers import AutoTokenizer

from optimized_sentence_transformers import HUB_PREFIX
from optimized_sentence_transformers.pooling import Pooling


class OptimizedSentenceTransformer:
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        modules: Any = None,
        device: Optional[Literal['cpu', 'gpu']] = None,
        cache_folder: Optional[Path] = None,
        use_auth_token: Union[bool, str, None] = None,
    ) -> None:
        if modules is not None:
            warnings.warn(
                '"modules" argument is not supported for now',
                UserWarning,
            )

        self._tokenizer = None
        self._session = None
        self._pooling = None

        self._cache_folder = self._get_cache_folder(cache_folder)
        self._provider = self._get_onnx_provider(device)
        self._model_path = self._download_model(
            model_name_or_path,
            use_auth_token,
        )

        self._load_model()

    def encode(
        self,
        sentences: Union[str, list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = 'sentence_embedding',
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        input_was_string = False

        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc='Batches', disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            out_features = self._encode_batch(sentences_batch)

            if output_value == 'token_embeddings':
                embeddings = []

                for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                    last_mask_id = len(attention) - 1

                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0:last_mask_id + 1])
            elif output_value is None:
                embeddings = []

                for sent_idx in range(len(out_features['sentence_embedding'])):
                    row = {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
            else:
                embeddings = out_features[output_value]

                if normalize_embeddings:
                    embeddings /= np.linalg.norm(
                        embeddings,
                        axis=1,
                    ).reshape((-1, 1))

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = np.asarray(all_embeddings)

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _encode_batch(
        self,
        sentences: Union[str, list[str]],
    ) -> dict[str, np.ndarray]:
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self._tokenizer.__dict__['model_max_length'],
            return_tensors='np',
        )
        outputs = self._session.run(
            None,
            dict(inputs),
        )

        return {
            'attention_mask': inputs['attention_mask'],
            'token_embeddings': outputs[0],
            'sentence_embedding': self._pooling(
                inputs['attention_mask'],
                outputs[0],  # token_embeddings
            ),
        }

    @staticmethod
    def _get_cache_folder(
        cache_folder: Optional[Path] = None,
    ) -> Path:
        if cache_folder is None:
            cache_folder = os.getenv('OPTIMIZED_SENTENCE_TRANSFORMERS_HOME')

            if cache_folder is None:
                cache_folder = Path.home() / '.cache' / 'optimized_sentence_transformers'
            else:
                cache_folder = Path(cache_folder)

        return cache_folder

    @staticmethod
    def _get_onnx_provider(
        device: Optional[str],
    ) -> str:
        if device is None:
            if ort.get_device() == 'GPU':
                provider = 'CUDAExecutionProvider'
            else:
                provider = 'CPUExecutionProvider'
        else:
            if device == 'cpu':
                provider = 'CPUExecutionProvider'
            elif device == 'gpu':
                provider = 'CUDAExecutionProvider'
            else:
                raise ValueError(f'unsupported device "{device}". Use "cpu", "gpu" or None')

        return provider

    def _download_model(
        self,
        model_name_or_path: Union[str, Path],
        use_auth_token: Union[bool, str, None],
    ) -> Path:
        if isinstance(model_name_or_path, Path):
            if model_name_or_path.exists():
                return model_name_or_path

            raise FileNotFoundError(f'path "{model_name_or_path}" does not exist')

        slash_count = model_name_or_path.count('/')

        if '\\' in model_name_or_path or slash_count > 1:
            raise ValueError(f'path "{model_name_or_path}" does not exist')

        if slash_count == 0:
            model_name_or_path = f'{HUB_PREFIX}/{model_name_or_path}'

        model_path = self._cache_folder / model_name_or_path

        if not model_path.exists():
            snapshot_download(
                model_name_or_path,
                cache_dir=self._cache_folder,
                use_auth_token=use_auth_token,
            )

        return model_path

    def _load_model(
        self,
    ) -> None:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            do_lower_case=True,
            fast=True,
        )
        self._pooling = Pooling.from_config(self._model_path / 'pooling.json')
        self._session = ort.InferenceSession(
            str(self._model_path / 'model.onnx'),
            sess_options,
            providers=[self._provider],
        )
