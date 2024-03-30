from pathlib import Path
from typing import Union, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
from peft import LoraModel, LoraConfig, set_peft_model_state_dict, load_peft_weights
import gc

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
hf_access_token=""

class WeightInterpolationGeneration: 
    STOP_TOKEN = "</s>"

    def __init__(
        self, 
        base_model: Union[str, Path],
        antiexpert_model: Union[str, Path] = None,
        expert_model: Union[str, Path] = None, 
        seed: int = 42,
        peft: bool = False,
        alpha: float = 1.0,
    ):
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        gc.collect()

        self.peft = peft

        if not peft:
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model).to(self.device)
        if peft:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False
            )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant_config,
                token=hf_access_token,
                device_map="auto",
            )

        torch.cuda.empty_cache()

        if expert_model and antiexpert_model:
            with torch.no_grad():
                base_state_dict = {}
                if not peft:
                    expert_dict = AutoModelForCausalLM.from_pretrained(expert_model).to(self.device).state_dict()
                    antiexpert_dict = AutoModelForCausalLM.from_pretrained(antiexpert_model).to(self.device).state_dict()
                else:
                    expert_dict = load_peft_weights(expert_model, device=self.device)
                    antiexpert_dict = load_peft_weights(antiexpert_model, device=self.device)
                
                alpha = torch.tensor(alpha).to(self.device)
                
                if not peft:
                    for key in expert_dict:
                        base_state_dict[key] = alpha * expert_dict[key] + (1 - alpha) * antiexpert_dict[key]

                    del expert_dict
                    del antiexpert_dict
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.base_model.load_state_dict(base_state_dict, strict=False)
                else:
                    # changes:
                    peft_model_path = expert_model

                    peft_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=peft_model_path)
                    peft_config.inference_mode = True
                    self.base_model = LoraModel(self.base_model, peft_config, adapter_name="expert")
                    base_state_dict = {}

                    for key in expert_dict:
                        base_state_dict[key[11:]] = alpha * expert_dict[key] if "lora_A" in key else expert_dict[key]
                    

                    set_peft_model_state_dict(self.base_model, base_state_dict, adapter_name="expert")

                    del expert_dict
                    del base_state_dict
                    torch.cuda.empty_cache()
                    gc.collect()


                    peft_model_path = antiexpert_model

                    peft_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=peft_model_path)
                    peft_config.inference_mode = True
                    self.base_model.add_adapter(peft_config, adapter_name = "antiexpert")
                    base_state_dict = {}
                    for key in antiexpert_dict:
                        base_state_dict[key[11:]] = (1.0-alpha) * antiexpert_dict[key] if "lora_A" in key else antiexpert_dict[key]

                    set_peft_model_state_dict(self.base_model, base_state_dict, adapter_name="antiexpert")

                    del antiexpert_dict
                    del base_state_dict
                    torch.cuda.empty_cache()
                    gc.collect()
        
            if peft:
                self.base_model.set_adapter(["expert", "antiexpert"])
                print(self.base_model.active_adapters)
    
        self.base_model.eval()
        torch.cuda.empty_cache()
        gc.collect()

        tokenizer_path = expert_model if expert_model else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        assert self.tokenizer.eos_token_id == self.tokenizer.pad_token_id

    def __repr__(self):
        return f'<WeightInterpolationGenerator model_name_or_path="{self.model}">'
    
    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self,
                 prompt: Union[str, List[str]],
                 max_len: int = 20,
                 sample: bool = True,
                 k: int = 0,
                 p: float = 1.0,
                 temperature: float = 1.0,
                 **model_kwargs):
        if isinstance(prompt, str):
            prompt = [prompt]

        torch.cuda.empty_cache()
        gc.collect()
        
        encodings_dict = self.tokenizer.batch_encode_plus(prompt, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')

        input_ids = encodings_dict['input_ids'].to(self.device)
        attention_mask = encodings_dict['attention_mask'].to(self.device)
        batch_size, input_seq_len = input_ids.shape

        position_ids = attention_mask.cumsum(dim=1) - 1
        unfinished_sents = torch.ones(batch_size, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for step in range(max_len):

                ensemble_logits, _ = self.base_model(
                  input_ids, attention_mask=attention_mask, 
                  position_ids=position_ids, return_dict = False, **model_kwargs)
                
                if p < 1.0:
                    ensemble_logits = top_k_top_p_filtering(ensemble_logits, top_p=p)
                

                # in the first decoding step, we want to use the 'real' last position for each sentence
                if step == 0:
                    last_non_masked_idx = torch.sum(attention_mask, dim=1) - 1
                    next_token_logits = ensemble_logits[range(batch_size), last_non_masked_idx, :]
                else:
                    next_token_logits = ensemble_logits[:, -1, :]

                if sample:
                    # Temperature (higher temperature => more likely to sample low probability tokens)
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    if k > 0 or p < 1.0:
                        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=k, top_p=p)
                    # Sample
                    try:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                    except RuntimeError:
                        probs = F.softmax(torch.nan_to_num(next_token_logits), dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # either append a padding token here if <EOS> has been seen or append next token
                tokens_to_add = next_tokens * unfinished_sents + self.tokenizer.pad_token_id * (1 - unfinished_sents)

                # this updates which sentences have not seen an EOS token so far
                # if one EOS token was seen the sentence is finished
                eos_in_sents = tokens_to_add == self.tokenizer.eos_token_id
                unfinished_sents.mul_((~eos_in_sents).long())

                # stop when there is an EOS in each sentence
                if unfinished_sents.max() == 0:
                    break

                # Update input_ids, attention_mask and position_ids
                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

                del ensemble_logits
                del next_token_logits
                del next_tokens
                del tokens_to_add
                torch.cuda.empty_cache()
                gc.collect()

        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for output in input_ids[:, input_seq_len:]]
        return decoded_outputs