
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import librosa
import random
import copy

PAD_IDX = -9999

class simple_converter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sr = bundle.sample_rate
        self.labels = bundle.get_labels()
        self.num_chars = len(self.labels)
        self.ƒ = bundle.get_model().extract_features # Just want the feature extractor
        self.bundle = bundle
        self.linear = nn.Linear(768, self.num_chars) #29
        print("")
    
    def forward(self,x):
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


class simple_transformer(nn.Module):
    def __init__(self, nheads=1, d_model=768, num_layers=1) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sr = bundle.sample_rate
        self.labels = bundle.get_labels()
        #self.labels = list(self.labels)
        #self.labels.extend(" ")
        #self.labels = tuple(self.labels)

        self.num_chars = len(self.labels)
        self.ƒ = bundle.get_model().extract_features # Just want the feature extractor


        self.linear = nn.Linear(768,d_model) 
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,nhead=nheads,batch_first=True),
            num_layers=num_layers
        )

        self.char_out = nn.Linear(768,self.num_chars)
    
    def forward(self,x):
        mask = simple_transformer.create_mask(x)
        #x = self.linear(x)
        #x[x==PAD_IDX] = 0
        x = self.transformer(x,src_key_padding_mask = mask)
        x = self.char_out(x)
        return F.log_softmax(x, dim=-1)
    
    def create_mask(x, PAD_IDX = -9999):
        #x[:,100:] = PAD_IDX
        return ~(x!=PAD_IDX)[:,:,0] #false means use, true is ignore
        #return x==PAD_IDX


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

class simple_transformer_v2(nn.Module):
    def __init__(self, nheads=4, d_model=768, num_layers=4) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sr = bundle.sample_rate
        self.labels = bundle.get_labels()
        self.num_chars = len(self.labels)
        self.ƒ = bundle.get_model().extract_features # Just want the feature extractor


        self.linear = nn.Linear(768,d_model) 
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,nhead=nheads,batch_first=True),
            num_layers=num_layers
        )

        self.char_out = nn.Linear(768,self.num_chars)
    
    def forward(self,x):
        mask = simple_transformer.create_mask(x)
        #x = self.linear(x)
        #x[x==PAD_IDX] = 0
        x = self.transformer(x,src_key_padding_mask = mask)
        x = self.char_out(x)
        return F.log_softmax(x, dim=-1)
    
    def create_mask(x, PAD_IDX = -9999):
        #x[:,100:] = PAD_IDX
        return ~(x!=PAD_IDX)[:,:,0] #false means use, true is ignore
        #return x==PAD_IDX


class simple_transformer_v3(nn.Module):
    def __init__(self, nheads=4, d_model=768, num_layers=4) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.bundle = bundle
        self.sr = bundle.sample_rate
        self.labels = bundle.get_labels()
        self.num_chars = len(self.labels)
        self.ƒ = bundle.get_model().extract_features # Just want the feature extractor

        
        self.linear = nn.Linear(768,d_model) 
            
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model,nhead=nheads,batch_first=True),
            num_layers=num_layers
        )

        self.char_out = nn.Linear(768,self.num_chars)

    
    def forward(self,x):
        mask = simple_transformer_v3.create_mask(x)

        x = self.transformer(x,src_key_padding_mask = mask)
        x = self.char_out(x)
        return F.log_softmax(x, dim=-1)
    
    def create_mask(x, PAD_IDX = -9999):
        #x[:,100:] = PAD_IDX
        return ~(x!=PAD_IDX)[:,:,0] 
        

def collect_and_preprocess_data():
    flac_files = []
    labels_files = []

    for dir in os.listdir("/kaggle/input/librispeech-clean/LibriSpeech/train-clean-360"):
        for file in os.listdir(os.path.join("/kaggle/input/librispeech-clean/LibriSpeech/train-clean-360",dir)):
            for file_d in os.listdir(os.path.join("/kaggle/input/librispeech-clean/LibriSpeech/train-clean-360",dir,file)):
                if file_d.endswith(".flac"):
                    flac_files.append(os.path.join("/kaggle/input/librispeech-clean/LibriSpeech/train-clean-360",dir,file,file_d))
                if file_d.endswith(".txt"):
                    labels_files.append(os.path.join("/kaggle/input/librispeech-clean/LibriSpeech/train-clean-360",dir,file,file_d))
            
    dirs = []
    all_labels = []
    for labels in labels_files:
        #data = pd.read_csv(f'{labels}')
        #df = pd.concat([df,data])
        with open(labels, "r") as file:
            lines = file.readlines()

        lines = [line.strip("\n") for line in lines]
        for j in range(len(lines)):
            dirs.append(labels)
        all_labels.extend(lines)
    ids = [x.split(" ")[0] for x in all_labels]
    true_labels = [" ".join(x.split(" ")[1:]) for x in all_labels]

    flac_dict = {flac.split("/")[-1].split(".flac")[0]: flac for flac in flac_files}

    organized_flac = []
    for id in tqdm(ids):
        if id in flac_dict:
            organized_flac.append(flac_dict[id])
        else:
            print("ERROR")
    df = pd.DataFrame()
    df["dir"] = organized_flac
    df["ids"] = ids
    df["labels"] = true_labels

    train, valid = train_test_split(df, test_size=0.2)

    return train,valid

def add_noise(waveform_np):
    noise_factor = random.uniform(0.002, 0.005)
    noise = np.random.randn(len(waveform_np))
    augmented_audio = waveform_np + noise_factor * noise
    return augmented_audio / (augmented_audio.max() + 1e-10)  # Normalize

def change_volume(waveform_np):
    min_gain = 0.7
    max_gain = 1.3
    gain = np.random.uniform(min_gain, max_gain)
    return waveform_np * gain

def change_speed(waveform_np):
    speed = random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(waveform_np, rate=speed)

def pitch_shift(waveform_np, sample_rate):
    n_steps = random.randint(-2, 2)
    return librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)

def time_mask(waveform_np, mask_factor=0.2):
    mask_start = np.random.randint(0, len(waveform_np) - int(mask_factor * len(waveform_np)))
    waveform_np[mask_start:mask_start + int(mask_factor * len(waveform_np))] = 0
    return waveform_np
    
def augment_audio(waveform, sample_rate=16000, augmentations=None, mask_factor=0.2):
    waveform_np = waveform.numpy()  # Convert tensor to numpy array

    if augmentations:
        for augmentation_type in augmentations:
            if augmentation_type == "add_noise":
                waveform_np = add_noise(waveform_np)
            elif augmentation_type == "change_speed":
                waveform_np = change_speed(waveform_np)
            elif augmentation_type == "pitch_shift":
                waveform_np = pitch_shift(waveform_np, sample_rate)
            elif augmentation_type == "time_mask":
                waveform_np = time_mask(waveform_np, mask_factor)
            elif augmentation_type == "change_volume":
                waveform_np = change_volume(waveform_np)

    # Convert back to tensor with the correct data type
    waveform = torch.tensor(waveform_np, dtype=torch.float)

    return waveform

class DataSet(Dataset):
    def __init__(self, local_df, encoder, decoder, augmentations=None):
        self.df = local_df
        self.sr = decoder.sr
        self.decoder = decoder
        self.encoder = encoder
        self.augmentations = augmentations 

    def __getitem__(self, index):
        dir = self.df.iloc[index].dir
        ids = self.df.iloc[index].ids
        labels = self.df.iloc[index].labels
        labels = labels.replace("","-")
        labels = labels.replace(" ","|")
        #print(labels)
        char_to_idx = {char: idx for idx, char in enumerate(self.decoder.labels)}
        ground_truth_token = torch.tensor([char_to_idx[c] for c in labels])

        waveform, sample_rate = torchaudio.load(dir)

        if sample_rate != self.decoder.sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.decoder.sr)
            # Ensure the sample rate expected by the encoder

        if self.augmentations:
            waveform = augment_audio(waveform, sample_rate=self.decoder.sr, augmentations=self.augmentations)
        
        waveform = waveform.to(dtype=torch.float) 

        with torch.no_grad():
            latent, _ = self.encoder(waveform)
        
        latent = latent[-1].squeeze()
        return latent, ground_truth_token
        
    def __len__(self):
        return len(self.df)
    

def collate_fn(batch): # stack batchwise
    latents, labels = zip(*batch)

    latents_padded = pad_sequence(latents, batch_first=True, padding_value=PAD_IDX)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return latents_padded, labels_padded


def train(decoder, augmentation, batch_size = 64, learning_rate = 0.0001, MAX_EPOCHS = 50, save=True):
    train, valid = collect_and_preprocess_data()
    train_ds = DataSet(train,decoder.ƒ,decoder,augmentations=augmentation)
    valid_ds = DataSet(valid,decoder.ƒ,decoder)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) #16
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) #16

    optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)#,weight_decay = 0.000000001
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)   

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    print("Start training")
    train_losses = []
    valid_losses = []


    for epoch in range(MAX_EPOCHS):
        train_counter = 0
        train_running_loss = 0
        valid_counter = 0
        valid_running_loss = 0 
        for j, (latents, labels) in enumerate(tqdm(train_dl)):
            decoder.train()
    
            labels = labels.squeeze()
            optim.zero_grad()
            latents = latents.to(device)
            labels = labels.to(device)
            prediction = decoder(latents)

            prediction = prediction.permute(1, 0, 2)  
            
            gt_string = "".join([decoder.labels[idx] for idx in labels[0]])
            gt_string = gt_string.replace("-", "").replace("|", " ")
            
            target_lengths = torch.tensor([len(lbl[lbl != -1]) for lbl in labels], dtype=torch.long).to(device)
            #input_lengths = torch.tensor([prediction.size(0)] * prediction.size(1), dtype=torch.long).to(device)
            input_lengths = torch.tensor([l[l!=PAD_IDX].view(-1,768).shape[0] for l in latents]).to(device)
            #flattened_labels = torch.tensor([lbl[lbl != -1] for lbl in labels])

            loss = criterion(prediction, labels, input_lengths, target_lengths)
         
            
            train_counter += 1
            train_running_loss += loss.item()

            loss.backward()
            optim.step()

            greedyDecoder = GreedyCTCDecoder(labels=decoder.labels)
            result_str = greedyDecoder(prediction[:,0]).replace("|"," ")
            print(f"Iteration {epoch}, Batch {j}, Predicted: {result_str}, truth : {gt_string}")
            

        for j, (latents, labels) in enumerate(tqdm(valid_dl)):
            with torch.no_grad():
                decoder.eval()
        
                labels = labels.squeeze()
                latents = latents.to(device)
                labels = labels.to(device)
                prediction = decoder(latents)

                greedyDecoder = GreedyCTCDecoder(labels=decoder.labels)
                result_str = greedyDecoder(prediction[0]).replace("|"," ")
        
                prediction = prediction.permute(1, 0, 2)  
        
                
                gt_string = "".join([decoder.labels[idx] for idx in labels[0]])
                gt_string = gt_string.replace("-", "").replace("|", " ")
                
                target_lengths = torch.tensor([len(lbl[lbl != -1]) for lbl in labels], dtype=torch.long).to(device)
                #input_lengths = torch.tensor([prediction.size(0)] * prediction.size(1), dtype=torch.long).to(device)
                input_lengths = torch.tensor([l[l!=PAD_IDX].view(-1,768).shape[0] for l in latents]).to(device)
                #flattened_labels = torch.tensor([lbl[lbl != -1] for lbl in labels])
        
                loss = criterion(prediction, labels, input_lengths, target_lengths)
        
            
                print(f"Iteration {epoch}, Batch {j}, Predicted: {result_str}, truth : {gt_string}")

                valid_counter += 1
                valid_running_loss += loss.item()
                
        train_losses.append(train_running_loss / train_counter)
        valid_losses.append(valid_running_loss / valid_counter)

        plt.clf()
        plt.plot(train_losses)
        plt.plot(valid_losses,color='red')
        plt.savefig("loss_curves")

        if save:
            torch.save({"state_dict" : copy.deepcopy(decoder.state_dict()),
                    "train_losses" : train_losses,
                    "valid_losses" : valid_losses},
                    f"{epoch}_model_linear_ft")


if __name__ == "__main__":
    decoder =  simple_converter()#simple_transformer_v3() #simple_transformer() #simple_converter()
    augmentation=["add_noise", "change_volume", "pitch_shift"]
    train(decoder,augmentation) # Use rest of defaults 