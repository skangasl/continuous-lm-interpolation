import json
import logging
import math
from pathlib import Path
from typing import Iterable, Union, TypeVar, Iterable, List
import pandas as pd
from tqdm.auto import tqdm

from generation.weight_interpolation_generation import WeightInterpolationGeneration
from generation.weight_interpolation_generation_multidim import WeightInterpolationGenerationMultiDimensional
T = TypeVar('T')

logging.disable(logging.CRITICAL)  # Disable logging from transformers

def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def _generation_helper(prompts: pd.Series,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: Union[WeightInterpolationGeneration, WeightInterpolationGenerationMultiDimensional],
                 out_file: Path,
                 **generate_kwargs):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Generate with prompts
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'Generation',
                       dynamic_ncols=True,
                       postfix={'batch_size': batch_size}):
        # Generate
        batch = generator.generate(prompt, max_len, **generate_kwargs)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def weight_interpolation(prompts: pd.Series,
             max_len: int,
             num_samples: int,
             batch_size: int,
             model_name_or_path: str,
             expert_name_or_path: str,
             antiexpert_name_or_path: str,
             out_file: Path,
             peft: bool,
             alpha: float,
             **generate_kwargs) -> Iterable[str]:

    generator = WeightInterpolationGeneration(
        base_model=model_name_or_path, 
        expert_model=expert_name_or_path,
        antiexpert_model=antiexpert_name_or_path,
        peft=peft,
        alpha=alpha
    )

    yield from _generation_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs
    )

def weight_interpolation_multidimensional(prompts: pd.Series,
             max_len: int,
             num_samples: int,
             batch_size: int,
             model_name_or_path: str,
             expert_model_dict_path: str,
             interpolation_parameter_dict_path: str,
             out_file: Path,
             peft: bool,
             **generate_kwargs) -> Iterable[str]:

    with open(expert_model_dict_path) as json_file:
        expert_model_dict = json.load(json_file)
    
    with open(interpolation_parameter_dict_path) as json_file:
        interpolation_param_dict = json.load(json_file)

    generator = WeightInterpolationGenerationMultiDimensional( 
        base_model=model_name_or_path,
        expert_model_dict=expert_model_dict,
        interpolation_parameter_dict=interpolation_param_dict,
        peft=peft
    )

    yield from _generation_helper(
        prompts=prompts,
        max_len=max_len,
        num_samples=num_samples,
        batch_size=batch_size,
        generator=generator,
        out_file=out_file,
        **generate_kwargs
    )