
import optuna
import torch
import torch.nn as nn
import random
import numpy as np
from transformers import T5Tokenizer
from train_r import train_model, CoeditDataset, TransFormerModel, TransFormerModel_v2
from torch.utils.data import DataLoader
from itertools import islice
from datasets import load_dataset

# Set Seeds for reproducibility
seed = 577
random.seed(seed)  # Python random module seed
np.random.seed(seed) # numpys random module seed
torch.manual_seed(seed) # torch random module seed
 
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 50

dataset = load_dataset("grammarly/coedit", split="train", streaming=True)

# Split the dataset using islice
train_data = list(islice(dataset, 0, 16000))  # First 8k for training
val_data = list(islice(dataset, 16000, 18000))  # Next 1k for validation
test_data = list(islice(dataset, 18000, 20000))  # Last 1k for testing

#train_data = list(islice(dataset, 0, 16000))  
#val_data = list(islice(dataset, 16000, 18000))  
#test_data = list(islice(dataset, 18000, 20000)) 

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1e-3) # Since we are fine-tuning, we want small learning rates
    batch_size = trial.suggest_categorical("batch_size", [8,16,32,64,128])
    optimizer_str = trial.suggest_categorical("optimizer_str",["Adam","SGD","AdamW"])
    l2_reg = trial.suggest_loguniform("l2_reg", 1e-15, 1e-4)
    d_model = trial.suggest_categorical('d_model',[512,768,1024])
    nhead = trial.suggest_categorical('nhead',[1,2,4,8,12])
    nlayers = trial.suggest_categorical('nlayers',[1,2,4,6,8,10])
    #dim_feedforward = trial.suggest_categorical('dim_feedforward',[512,768,1024])

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    #model = TransFormerModel(tokenizer.vocab_size,d_model,nhead,nlayers,nlayers,dim_feedforward,128,tokenizer.pad_token_id).to(DEVICE)
    model = TransFormerModel_v2(tokenizer.vocab_size,tokenizer.pad_token_id,128,d_model,nhead,nlayers).to(DEVICE)

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

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) #-1 for t5

    hyp= {
            "learning_rate" : learning_rate,
            "batch_size" : batch_size,
            "optimizer_str" : optimizer_str,
            "l2_reg" : l2_reg,
            "d_model" : d_model,
            "nhead" : nhead,
            "nlayers"  : nlayers,
            #"dim_feedforward" : dim_feedforward
    }

    output, model = train_model(model,train_loader,val_loader,optimizer,scheduler,criterion,MAX_EPOCHS)

    val_loss = np.min(output['valid_loss'])

    try:
        if val_loss < study.best_value:
            torch.save({"state_dict":model.state_dict(),'hyp' : hyp, 'val_loss' : output['valid_loss'], 'train_loss' : output['train_loss']}, "best_model_transformer_v2.torch") 
            #model.save_pretrained("t5_fine_tuned_optuna")
            #tokenizer.save_pretrained("t5_fine_tuned_optuna")
    except:
        torch.save({"state_dict":model.state_dict(),'hyp' : hyp, 'val_loss' : output['valid_loss'], 'train_loss' : output['train_loss']}, "best_model_transformer_v2.torch") 
        #model.save_pretrained("t5_fine_tuned_optuna_2")
        #tokenizer.save_pretrained("t5_fine_tuned_optuna_2")
        
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    df = study.trials_dataframe()

    df.to_csv("optuna_study_res_3.csv", index =False)