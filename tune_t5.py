
import optuna
import torch
import torch.nn as nn
import random
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from train_r import train_model, CoeditDataset
from torch.utils.data import DataLoader
from itertools import islice
from datasets import load_dataset

# HP tuning the T5 model.

# Set Seeds for reproducibility
seed = 577
random.seed(seed)  # Python random module seed
np.random.seed(seed) # numpys random module seed
torch.manual_seed(seed) # torch random module seed
 
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 15

dataset = load_dataset("grammarly/coedit", split="train", streaming=True)

# Split the dataset using islice
# note that this data is already shuffled so this is somewhat justified
train_data = list(islice(dataset, 0, 8000))  # First 8k for training
val_data = list(islice(dataset, 8000, 9000))  # Next 1k for validation
test_data = list(islice(dataset, 9000, 10000))  # Last 1k for testing

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1e-3) # Since we are fine-tuning, we want small learning rates
    batch_size = trial.suggest_categorical("batch_size", [8,16,32])
    optimizer_str = trial.suggest_categorical("optimizer_str",["Adam","SGD","AdamW"])
    l2_reg = trial.suggest_loguniform("l2_reg", 1e-15, 1e-4)
    base_model = trial.suggest_categorical("base_model",["t5-small","t5-base"]) #t5 large cant fit in memory :(

    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model).to(DEVICE)

    train_dataset = CoeditDataset(train_data, tokenizer)
    val_dataset = CoeditDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if optimizer_str == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    elif optimizer_str == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg, momentum=0.9)
    elif optimizer_str == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    else:
        raise NotImplementedError("Not implemented yet")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=MAX_EPOCHS,
        pct_start=0.1
    )

    hyp = {
        "learning_rate" : learning_rate,
        "batch_size" : batch_size,
        "l2_reg" : l2_reg,
        "optimizer_str" : optimizer_str,
        "base_model" : base_model
    }

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


    output, model = train_model(model,train_loader,val_loader,optimizer,scheduler,criterion,MAX_EPOCHS)

    val_loss = np.min(output['valid_loss'])

    try:
        if val_loss < study.best_value:
            #torch.save({"state_dict":model.state_dict(), "hyp" : hyp}, "best_model.torch") 
            model.save_pretrained("t5_fine_tuned_optuna")
            tokenizer.save_pretrained("t5_fine_tuned_optuna")
    except:
        #torch.save({"state_dict":model.state_dict(), "hyp" : hyp}, "best_model.torch") 
        model.save_pretrained("t5_fine_tuned_optuna_2")
        tokenizer.save_pretrained("t5_fine_tuned_optuna_2")
        
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    df = study.trials_dataframe()

    df.to_csv("optuna_study_res_2.csv", index =False)