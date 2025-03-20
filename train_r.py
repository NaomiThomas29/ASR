# Install necessary packages
#!pip install datasets transformers torch nltk jiwer
from torcheval.metrics.functional import word_error_rate

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from copy import deepcopy
import os
from itertools import islice
#from google.colab import drive
import torch.nn as nn

# Mount Google Drive
#drive.mount('/content/drive')

# Hyperparameters
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE =0.0001
EPOCHS = 100
MODEL_SAVE_PATH = "/content/drive/MyDrive/t5_fine_tuned.pt"
    

class TransFormerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_len, pad_token_id):
        super(TransFormerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.positional_encoding = nn.Parameter(torch.ones(1, max_seq_len, d_model))
        # want one parameter per seq len on the dimensionaly of the model
        # start a position encoding learning paramer

        self.transformer = nn.Transformer(
            d_model= d_model,
            nhead= nhead,
            num_encoder_layers =num_encoder_layers,
            num_decoder_layers =num_decoder_layers,
            dim_feedforward = dim_feedforward,
            batch_first=True  # its just easier for my intuition to use batch first
        )
        # Project to our vocab size and thus a probability distriburion-like distribution (its not softmaxed here because CE loss needs logits)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask, labels):
        # get embeddings. For both make sure the size is applicible for the position encoding
        src_embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        tgt_embeddings = self.embedding(labels) + self.positional_encoding[:, :labels.size(1), :]

        # can use attention mask we already have, just note that we have to complement it because of the way transformers handle it
        src_key_padding_mask = ~attention_mask.bool()

        encoding = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer.decoder(tgt_embeddings,encoding,tgt_key_padding_mask=src_key_padding_mask)

        # Project to vocabulary size
        logits = self.output_layer(output)
        return logits
    
class TransFormerModel_v2(nn.Module):
    def __init__(self, vocab_size, pad_token_id,max_seq_len=128, d_model = 768, nheads = 4, nlayers = 2):
        super(TransFormerModel_v2, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=pad_token_id)
        self.positional_encoding = nn.Parameter(torch.ones(1, max_seq_len, self.d_model))
        # want one parameter per seq len on the dimensionaly of the model
        # start a position encoding learning paramer

        self.transformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            self.d_model,
            nhead=nheads,
            batch_first=True
        ),
        num_layers=nlayers)

        self.output_layer = nn.Linear(self.d_model, vocab_size)

    def forward(self, input_ids, attention_mask, labels):
        # get embeddings. For both make sure the size is applicible for the position encoding
        src_embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]

        # can use attention mask we already have, just note that we have to complement it because of the way transformers handle it
        src_key_padding_mask = ~attention_mask.bool()

        encoding = self.transformerEncoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)

        logits = self.output_layer(encoding + src_embeddings)
        return logits




# Preprocessing function
def remove_prompt(text):
    """Remove the prompt (everything before and including the colon) from the text."""
    prompt_end = text.find(":")
    if prompt_end != -1:
        return text[prompt_end + 1:].strip()
    else:
        return text.strip()

# Dataset class for tokenization and preparation
class CoeditDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        stripped_input = remove_prompt(example["src"])
        input_text = "Make this sentence grammatically correct: " + stripped_input
        target_text = example["tgt"]

        # Tokenize input and output
        input_tokens = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        target_tokens = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )

        return {
            "input_ids": input_tokens["input_ids"].squeeze(),
            "attention_mask": input_tokens["attention_mask"].squeeze(),
            "labels": target_tokens["input_ids"].squeeze(),
        }


# Training loop
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs):
    valid_lossess = []
    train_lossess = []
    current_best_val_loss = 9999
    best_model_copy = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            ##T5
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ##loss = outputs.loss

            ##custom
            #logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            logits = outputs 

            flattened_logits = logits.view(-1, logits.size(-1))  
            flattened_labels = labels.view(-1) 

            flattened_labels = flattened_labels.to(flattened_logits.device)

            # Compute the loss
            loss = criterion(flattened_logits, flattened_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


        valid_lossess.append(validate_model(model, val_loader, criterion))
        train_lossess.append(total_loss / len(train_loader))

        if valid_lossess[-1] > 5.3 and epoch > 20:
            print("Early stop because of poor performance")
            return {"train_loss" : train_lossess, "valid_loss" : valid_lossess}, best_model_copy

        if valid_lossess[-1] < current_best_val_loss:
            torch.save(deepcopy(model.state_dict()),"BEST_TRANS_NEW.torch")
            current_best_val_loss = valid_lossess[-1]
            best_model_copy = deepcopy(model)

    
    return {"train_loss" : train_lossess, "valid_loss" : valid_lossess}, best_model_copy


# Validation loop
def validate_model(model, val_loader, criterion):
    model.eval() # model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ##loss = outputs.loss

            ##custom
            #logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            logits = outputs 

            flattened_logits = logits.view(-1, logits.size(-1))  
            flattened_labels = labels.view(-1) 

            flattened_labels = flattened_labels.to(flattened_logits.device)

            # Compute the loss
            loss = criterion(flattened_logits, flattened_labels)

            total_loss += loss.item()

    print(f"Validation Loss: {total_loss/len(val_loader):.4f}")

    return total_loss/len(val_loader)


# Evaluate and print sample outputs
def evaluate_on_test_data(model, test_data, tokenizer):
    model.eval()
    print("\nSample Outputs After Fine-Tuning:\n")

    total_wer=0

    for count, example in enumerate(test_data):

        stripped_input = remove_prompt(example["src"])
        input_text = "Make this sentence grammatically correct: " + stripped_input
        target_text = example["tgt"]

        ##temp
        def preproc_error_correction(example):
            def remove_prompt(text):
                """Remove the prompt (everything before and including the colon) from the text."""
                prompt_end = text.find(":")
                if prompt_end != -1:
                    return text[prompt_end + 1:].strip()
                else:
                    return text.strip()
            stripped_input = remove_prompt(example['input'])
            #input_text = "Make this sentence grammatically correct: " + stripped_input
            input_text = stripped_input
            target_text = "NOT RELEVANT FOR TESTING"
            input_tokens = tokenizer(
                input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
            )

            target_tokens = tokenizer(
                target_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
            )


            return {
                "input_ids": input_tokens["input_ids"].squeeze(),
                "attention_mask": input_tokens["attention_mask"].squeeze(),
                "labels": target_tokens["input_ids"].squeeze(),
            }
        
        encoded_str = preproc_error_correction({"input":input_text})

        ##temp

        # Tokenize input
        input_ids = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding="max_length"
        ).input_ids

        attention_mask = torch.tensor(tokenizer(input_text)['attention_mask'])
        model.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        # Generate output
        with torch.no_grad():
            try:
                outputs = model.generate(input_ids, max_length=MAX_LENGTH, num_beams=5, early_stopping=True)
            except:
                outputs = model(encoded_str['input_ids'].unsqueeze(0).to(DEVICE),encoded_str['attention_mask'].unsqueeze(0).to(DEVICE),encoded_str['input_ids'].unsqueeze(0).to(DEVICE)) # last input_ids is just so it computed well, it doesnt have a real effect
                outputs = torch.argmax(outputs,-1)


        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Print input, target, and prediction
        print(f"Sample {count + 1}:")
        print(f"Input: {example['src']}")
        print(f"Target: {target_text}")
        # print(f"Prediction: {prediction}\n")

        # Compute WER
        sample_wer = word_error_rate(target_text, prediction)
        total_wer += sample_wer

        # Print input, target, prediction, and WER
        print(f"Prediction: {prediction}")
        print(f"WER: {sample_wer:.4f}\n")

    avg_wer = total_wer / len(test_data)
    print(f"Average WER on Test Samples: {avg_wer:.4f}")


def evaluate_on_test_data_2(model, test_loader):
    model.eval() # model.eval()
    total_loss = 0
    total_wer = 0
    counter = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ##loss = outputs.loss

            ##custom
            #logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
            logits = outputs 

            output_str = tokenizer.decode(torch.argmax(outputs,-1)[0]).split(tokenizer.eos_token)[0]

            total_wer+=word_error_rate(output_str,tokenizer.decode(labels[0],skip_special_tokens=True))
            counter += 1

    print(f"total_wer: {total_wer / counter}")




# Main script
if __name__ == "__main__":


    # Load the Grammarly Coedit dataset
    dataset = load_dataset("grammarly/coedit", split="train", streaming=True)

    # Split the dataset using islice
    train_data = list(islice(dataset, 0, 15000))  # First 8k for training
    val_data = list(islice(dataset, 15000, 15000+4000))  # Next 1k for validation
    test_data = list(islice(dataset, 15000+4000, 15000+4000+1000))  # Last 1k for testing

    #TEMP
    tokenizer = T5Tokenizer.from_pretrained("/home/m33murra/test_vocal/error_correction/t5_fine_tuned_optuna_cp")
    #mdl = T5ForConditionalGeneration.from_pretrained("/home/m33murra/test_vocal/error_correction/t5_fine_tuned_optuna_cp")
    
    model_info = torch.load("/home/m33murra/test_vocal/TRANSNEW_336_0_SECOND_ONE")
    #hyp = model_info['hyp']
    #mdl =  #TransFormerModel(tokenizer.vocab_size,hyp['d_model'],hyp["nhead"],hyp['nlayers'],hyp['nlayers'],hyp['dim_feedforward'],128,tokenizer.pad_token_id)
    mdl = TransFormerModel_v2(tokenizer.vocab_size,tokenizer.pad_token_id,200,2048,8,10).to(DEVICE)
    mdl.load_state_dict(model_info)
    mdl.to(DEVICE)
    mdl.eval()

    test_dataset = CoeditDataset(test_data,tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)
    
    
    #evaluate_on_test_data(mdl.to(DEVICE),test_data,tokenizer)
    evaluate_on_test_data_2(mdl,test_loader) # for evaluation of the transformer model
    #TEMP


    # Load the tokenizer and T5 model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    #model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
    model = TransFormerModel_v2(tokenizer.vocab_size,tokenizer.pad_token_id,200,2048,8,10).to(DEVICE)

    # Prepare the datasets
    train_dataset = CoeditDataset(train_data, tokenizer)
    val_dataset = CoeditDataset(val_data, tokenizer)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0) #-1 for t5

    # Train the model
    train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=EPOCHS)
    
    # Save the trained model
    #model.save_pretrained("/content/drive/MyDrive/t5_fine_tuned")
    #tokenizer.save_pretrained("/content/drive/MyDrive/t5_fine_tuned")
    #print(f"Model and tokenizer saved to /content/drive/MyDrive/t5_fine_tuned")

    # Evaluate the model on test data
    evaluate_on_test_data(model, test_data, tokenizer)
