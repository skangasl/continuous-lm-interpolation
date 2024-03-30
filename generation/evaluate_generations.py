import argparse
import pandas as pd
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer
from finetuning.finetune_classifier import preprocess_text
from generation.weight_interpolation_generation import WeightInterpolationGeneration
from datasets import load_dataset
from tqdm import tqdm

hf_access_token=""

def perplexity(model, device='cuda'):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = model.tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = 4096 # model.base_model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model.base_model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        if not torch.isinf(torch.exp(neg_log_likelihood)).item():
            nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).nanmean())
    print(ppl)
    return ppl

def score_prediction(generations_df, model, tokenizer, n=3, device="cuda", max_len=512):
    outputs = [[] for _ in range(n)]
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating accuracy'):
        generations =  [row]
        for j, gen in enumerate(generations):
            # postprocessing
            text = preprocess_text(gen[0])

            inputs = tokenizer.encode_plus(
                text,
                None,
                truncation=True,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_token_type_ids=True
            )
            ids = torch.tensor([inputs['input_ids']], dtype=torch.long, device=device)
            mask = torch.tensor([inputs['attention_mask']], dtype=torch.long, device=device)
            token_type_ids = torch.tensor([inputs["token_type_ids"]], dtype=torch.long, device=device)
            if ids.size()[0] < max_len:
                output = model(ids, mask, token_type_ids, predict=True)
            else:
                output = np.nan

            if output.shape[1] > 1:
                output = output[:, -1]
            output = output.squeeze().item()
            
            outputs[i % n].append(output)

    return np.nanmean(np.array(outputs), axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--generations_file", help='a jsonl file with generations and attribute scores', type=str)
    parser.add_argument("-d", "--eval_dims", help="list of evaluation dimensions", nargs='+')
    parser.add_argument("-m", "--eval_model_files", help="list of evaluation model directories", nargs='+')
    parser.add_argument("-n", "--num_prompts", help='number of generations per prompt', type=int)
    parser.add_argument("-a", "--alpha", help='alpha parameter for interpolation', type=float)
    args = parser.parse_args()
    generations_file = args.generations_file
    eval_dims = args.eval_dims
    eval_model_files = args.eval_model_files
    n = args.num_prompts
    alpha = args.alpha
    max_len = 32

    print(generations_file)
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_json(generations_file, lines=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # compute perplexity
    if "perplexity" in eval_dims:
        base_model = eval_model_files[0]
        pos_model = eval_model_files[1]
        neg_model = eval_model_files[2]
        model = WeightInterpolationGeneration(
            base_model = base_model, antiexpert_model=neg_model, 
            expert_model=pos_model, peft=True, alpha=alpha)
        

        torch.cuda.empty_cache()
        with torch.no_grad():
            ppl = perplexity(model, device=device)

        with open(output_dir / 'perplexity_results.txt', 'w') as fo:
            fo.write(f'perplexity = {ppl}\n')

    else:
        # compute scores for all classes
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', 
                                                    truncation=True, 
                                                    do_lower_case=True,
                                                    add_special_tokens=True,
                                                    padding='max_length',
                                                    max_length=max_len)
        tokenizer.do_lower_case = True
        out_dct = {}
        for i, dim in enumerate(eval_dims):
            model = torch.load(eval_model_files[i]).to(device)
            # if 
            model.eval()
            with torch.no_grad():
                scores = score_prediction(generations_df, model, tokenizer, device=device, n=n, max_len=max_len)
            
            out_dct[dim] = scores
        print([f"{x}: {np.nanmean(out_dct[x])}" for x in out_dct])
        out_df = pd.DataFrame(out_dct)
        out_df.to_csv(output_dir / 'eval_results.csv')


if __name__ == '__main__':
    main()