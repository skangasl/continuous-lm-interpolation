import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import cuda
from datasets import load_dataset
import click
from typing import Optional
import os
import re

def preprocess_text(text):
    punc='''0-9!;:'",.?'''
    text = re.sub(r'[^a-zA-Z'+punc+']', ' ',text)
    text = text.lower()
    text = " ".join(text.split())
    return text

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = preprocess_text(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClass(torch.nn.Module):
    def __init__(self, model_type="regressor"):
        super(RobertaClass, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", 
            output_hidden_states=True, 
            num_labels=1
        )
        self.model_type = model_type
        self.calibrator = None

    def forward(self, input_ids, attention_mask, token_type_ids, predict=False):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask).logits
        try:
            if predict and self.model_type == "classifier":
                output = torch.sigmoid(output)
            if self.calibrator:
                output = self.calibrator.predict(output[:, -1])
        except AttributeError:
            pass
        return output 
        

def train(epoch, model, optimizer, training_loader, validation_loader, loss_function, device, grad_clip_val=1, output_dir=None, sqrt=False):
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        if len(targets.size()) == 1:
            targets = targets.unsqueeze(1)

        outputs = model(ids, mask, token_type_ids).to(device)
        loss = loss_function(outputs, targets)
        if sqrt:
            loss = torch.sqrt(loss)
        tr_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        optimizer.step()

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _%100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training Loss per 100 steps: {loss_step} Curr Loss: {loss.item()}")
            val_loss = valid(model, validation_loader, loss_function, device, sqrt=sqrt)
            print(f"Validation Loss Epoch {epoch} step {_}: {val_loss}")
            model.train()

            if output_dir is not None and _%300==0:
                torch.save(model, f"{output_dir}/pytorch_roberta_epoch_{epoch}_step_{_}.bin")

    epoch_loss = tr_loss/nb_tr_steps
    print(f"Training Loss Epoch {epoch}: {epoch_loss}")
    return 

def valid(model, testing_loader, loss_function, device, sqrt=False):
    model.eval()
    tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            if len(targets.size()) == 1:
                targets = targets.unsqueeze(1)

            outputs = model(ids, mask, token_type_ids).to(device)
            loss = loss_function(outputs, targets)
            if sqrt:
                loss = torch.sqrt(loss)
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

    epoch_loss = tr_loss/nb_tr_steps
    return epoch_loss

def finetune(data_file, out_dir, csv=True, loss="MSE", n_epoch=3, calibration=None, calibration_val = 0.05, pos_weight=-1):
    device = 'cuda' if cuda.is_available() else 'cpu'

    # Defining some key variables that will be used later on in the training
    # MAX_LEN = 128
    MAX_LEN = 32
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    
    LEARNING_RATE = 5e-06
    if "formality" in data_file:
        LEARNING_RATE = 1e-06
    if "sentiment" in data_file:
        LEARNING_RATE = 1e-5
    if "politeness" in data_file:
        LEARNING_RATE = 1e-7
        if loss == "BCE":
            LEARNING_RATE=5e-7
    if "simplicity" in data_file:
        LEARNING_RATE = 5e-7
    sqrt=False

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', 
                                                truncation=True, 
                                                do_lower_case=True,
                                                padding='max_length',
                                                max_length=MAX_LEN)
                                                 
    tokenizer.do_lower_case = True

    if csv:
        df = pd.read_csv(data_file).reset_index(drop=True)
        df = df.sample(frac=1)[:100000]
        train_data, val_data = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)[:1000]
    else:
        train_data = pd.DataFrame(load_dataset(data_file, split='train'))
        train_data = train_data[["sentence", "label"]]
        train_data.columns = ["text", "label"]
        train_data = train_data.sample(frac=1)[:40000]

        val_data = pd.DataFrame(load_dataset(data_file, split='validation'))[:1000]
        val_data = val_data[["sentence", "label"]]
        val_data.columns = ["text", "label"]
    
    if loss == "BCE" and data_file == "sst":
        train_data = train_data.loc[(train_data['label'] >= 0.7) | (train_data['label'] <= 0.3)]
        val_data = val_data.loc[(val_data['label'] >= 0.7) | (val_data['label'] <= 0.3)]
    elif loss == "BCE" and "politeness" in out_dir:
        train_data = train_data.loc[(train_data['label'] >= 0.9) | (train_data['label'] <= 0.1)]
        val_data = val_data.loc[(val_data['label'] >= 0.9) | (val_data['label'] <= 0.1)]

    if loss == "BCE":
        train_data["label"] = (train_data["label"] >= 0.5).astype(float)
        val_data["label"] = (val_data["label"] >= 0.5).astype(float)


    print(f"Model: {out_dir}")
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")

    
    if calibration == "soft_label" or calibration == "platt_and_soft":
        train_data["label"] = ((train_data["label"]) - calibration_val).abs()
    
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    training_set = SentimentData(train_data, tokenizer, MAX_LEN) #, n_class=n_labs)
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    training_loader = DataLoader(training_set, **train_params)

    validation_set = SentimentData(val_data, tokenizer, MAX_LEN) #, n_class=n_labs)
    validation_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    val_loader = DataLoader(validation_set, **validation_params)

    model_type = "classifier" if loss == "BCE" else "regressor"
    model = RobertaClass(model_type=model_type)
    model.to(device)

    # Creating the loss function and optimizer
    if loss == "MSE":
        loss_function = torch.nn.MSELoss()
    elif loss == "RMSE":
        loss_function = torch.nn.MSELoss()
        sqrt=True
    elif loss == "MAE":
        loss_function = torch.nn.L1Loss()
    else:
        if pos_weight > 0:
            pweight = torch.tensor([pos_weight]).to(device)
            loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pweight)
        else:
            loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay = 0.01)

    output_model_dir = f'{out_dir}'
    output_model_file = f'{out_dir}/pytorch_roberta.bin'
    output_vocab_file = f'{out_dir}/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for epoch in range(n_epoch):
        train(epoch, model, optimizer, training_loader, val_loader, loss_function, device, output_dir=output_model_dir, sqrt=sqrt)

        val_loss = valid(model, val_loader, loss_function, device, sqrt=sqrt)
        print(f"Validation Loss Epoch {epoch}: {val_loss}")
        torch.save(model, f"{output_model_dir}/pytorch_roberta_epoch_{epoch}.bin")
    
    torch.save(model, f"{output_model_file}")
    tokenizer.save_vocabulary(output_vocab_file)

@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str)
@click.option('--csv', default=False)
@click.option('--loss-function', required=False, type=str)
@click.option('--n-epoch', required=False, type=int, default=3)
@click.option('--calibration', required=False, type=str, default=None)
@click.option('--calibration_val', required=False, type=float, default=0.05)
@click.option('--pos_weight', required=False, type=float, default=-1)
def main(output_dir: str, dataset_file: Optional[str], csv: bool, loss_function: Optional[str], n_epoch: Optional[int], calibration: Optional[str], calibration_val: Optional[float], pos_weight: Optional[float]):
    finetune(dataset_file, output_dir, csv, loss_function, n_epoch=n_epoch, calibration=calibration, calibration_val = calibration_val, pos_weight=pos_weight)

if __name__ == '__main__':
    main()




