import json
from pathlib import Path
from typing import Union

import torch
from sentence_transformers import SentenceTransformer
from transformers import BertModel

_DYNAMIC_AXES = {  # noqa: WPS407
    0: 'batch_size',
    1: 'max_seq_len',
}
_MIN_ONNX_OPSET = 13
_DEFAULT_ONNX_OPSET = 17


def convert(
    model_name_or_path: Union[str, Path],
    out_path: Path,
    opset_version: int = _DEFAULT_ONNX_OPSET,
    **kwargs,
) -> None:
    if opset_version < _MIN_ONNX_OPSET:
        raise ValueError('opset_version must be >= 13 for converting SentenceTransformer to ONNX')

    out_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = SentenceTransformer(model_name_or_path)

    if not isinstance(model[0].auto_model, BertModel):
        raise ValueError('only BertModel based models are supported')

    if len(model) > 2:
        raise RuntimeError('models with additional normalization layer are not supported for now')

    example_inputs = model[0].tokenizer(
        ['hello'],
        padding=True,
        truncation=True,
        max_length=model[0].tokenizer.__dict__['model_max_length'],
        return_tensors='pt',
    )

    torch.onnx.export(
        model[0].auto_model,
        args=(
            example_inputs['input_ids'],
            example_inputs['attention_mask'],
            example_inputs['token_type_ids'],
        ),
        f=str(out_path / 'model.onnx'),
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[
            'input_ids',
            'attention_mask',
            'token_type_ids',
        ],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': _DYNAMIC_AXES,
            'token_type_ids': _DYNAMIC_AXES,
            'attention_mask': _DYNAMIC_AXES,
            'last_hidden_state': _DYNAMIC_AXES,
        },
        **kwargs,
    )

    model[0].tokenizer.save_pretrained(
        out_path,
        include_config=True,
    )

    with open(out_path / 'pooling.json', 'w', encoding='utf-8') as out:
        json.dump(
            {
                'word_embedding_dimension': model[1].word_embedding_dimension,
                'pooling_mode_cls_token': model[1].pooling_mode_cls_token,
                'pooling_mode_mean_tokens': model[1].pooling_mode_mean_tokens,
                'pooling_mode_max_tokens': model[1].pooling_mode_max_tokens,
                'pooling_mode_mean_sqrt_len_tokens': model[1].pooling_mode_mean_sqrt_len_tokens,
            },
            out,
            indent=4,
            ensure_ascii=False,
        )
