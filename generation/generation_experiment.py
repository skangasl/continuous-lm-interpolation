from pathlib import Path
from typing import Optional
import click
import pandas as pd
import torch
import os
from generation.generation import weight_interpolation, weight_interpolation_multidimensional

ALLOWED_MODELS = ['weight_interpolation', 'weight_interpolation_multidim']


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True, type=click.Choice(ALLOWED_MODELS))
@click.option('--pos-model', type=str, help="Positive model for DExperts")
@click.option('--neg-model', type=str, help="Negative model for DExperts")
@click.option('--expert-model-dict-path', type=str, help='Path to dictionary containing expert model paths')
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--max-prompts', default=-1, help='Max number of prompts to allow.')
@click.option('--batch-size', default=32)
@click.option('--resume/--no-resume', default=False)
@click.option('--alpha', default=0.0, help='Hyperparameter for ensemble methods')
@click.option('--interpolation-parameter-dict-path', type=str, help='Path to dictionary containing interpolation hyperparameters')
@click.option('--p', default=1.0, type=float, help='Hyperparameter for nucleus (top-p) sampling')
@click.option('--temperature', default=1.0, type=float, help='Hyperparameter for temperature in generation')
@click.option('--filter_p', default=0.9, type=float, help='Hyperparameter for truncation of p_base')
@click.option('--use_peft', default=False, help='Whether to use Parameter-Efficient Fine Tuning')
def main(output_dir: str, dataset_file: Optional[str], model: str, model_type: str, 
         pos_model: str, neg_model: str, expert_model_dict_path: Optional[str], n: int, max_tokens: int, max_prompts: int, batch_size: int, resume: bool,
         alpha: float, interpolation_parameter_dict_path: Optional[str], p: float, temperature: float, filter_p: float, use_peft: bool):
    
    # Load prompts from dataset file
    assert dataset_file.endswith('.jsonl')
    dataset = pd.read_json(dataset_file, lines=True)
    prompts = pd.json_normalize(dataset['prompt'])['text']
    if max_prompts > 0:
        prompts = prompts[:max_prompts]

    # Create output files
    output_dir = Path(output_dir)
    generations_file = output_dir / 'generations.jsonl'
    if not resume and os.path.exists(generations_file):
        return
    assert resume or not os.path.exists(generations_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup model for generation
    if model_type == 'weight_interpolation':
        weight_interpolation(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_name_or_path=pos_model,
            antiexpert_name_or_path=neg_model,
            out_file=generations_file,
            filter_p=filter_p,
            p=p,
            temperature=temperature,
            alpha=alpha,
            peft=use_peft,
        )
    elif model_type == 'weight_interpolation_multidim':
        weight_interpolation_multidimensional(
            prompts=prompts,
            max_len=max_tokens,
            num_samples=n,
            batch_size=batch_size,
            model_name_or_path=model,
            expert_model_dict_path=expert_model_dict_path,
            interpolation_parameter_dict_path=interpolation_parameter_dict_path,
            out_file=generations_file,
            filter_p=filter_p,
            p=p,
            temperature=temperature,
            peft=use_peft,
        )
    else:
        raise NotImplementedError(f'Model {model} not implemented')

    torch.cuda.empty_cache()
    print('Finished generation!')


if __name__ == '__main__':
    main()
