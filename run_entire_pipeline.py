
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import math
from torcheval.metrics.functional import word_error_rate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from torcheval.metrics.functional import word_error_rate

from transformers import T5ForConditionalGeneration, T5Tokenizer

class simple_converter_from_literature(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sr = bundle.sample_rate
        self.labels = bundle.get_labels()
        self.num_chars = len(self.labels)
        self.ƒ = bundle.get_model().extract_features # Just want the feature extractor
        self.bundle = bundle
        self.linear = nn.Linear(768, self.num_chars) #29
        self.linear.weight = bundle.get_model().aux.weight 
    
    def forward(self,x):
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


class TransFormerModel(nn.Module): # Deprecate
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
    
class GreedyCTCDecoder(torch.nn.Module): # From pytorch docs, sometimes useful for when certain charcaters are repeated
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1) 
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    
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
        self.linear.weight = bundle.get_model().aux.weight # Use weights from literature for the linear model and train the transformer decoder
        
        for param in self.linear.parameters():
            param.requires_grad = False
            
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
        
        return ~(x!=PAD_IDX)[:,:,0] 



class Pipeline:
    def __init__(self, encoder, asr_model, error_correction_model, tokenizer, extra_decoder = None,):
        self.encoder = encoder
        self.ASR_model = asr_model = asr_model
        self.error_correction_model = error_correction_model
        self.extra_decoder = extra_decoder  # such as greedy decoder
        self.tokenizer = tokenizer

        #Encoder is already frozen in evaluation mode
        self.ASR_model.eval().cpu()
        #self.error_correction_model.eval().cpu()
    
    def preproc_encoder(self,wav_files):
        output_wavs = []
        for wav in wav_files:
            waveform, sample_rate = torchaudio.load(wav)
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.ASR_model.sr)
            output_wavs.append(waveform)
        return output_wavs
    

    def preproc_error_correction(self,example):
        def remove_prompt(text):
            """Remove the prompt (everything before and including the colon) from the text."""
            prompt_end = text.find(":")
            if prompt_end != -1:
                return text[prompt_end + 1:].strip()
            else:
                return text.strip()
        stripped_input = remove_prompt(example['input'])
        if self.error_correction_model.base_model is None:
            input_text = stripped_input
            target_text = stripped_input
        else:
            input_text = "Make this sentence grammatically correct: " + stripped_input
            target_text = stripped_input
        #input_text = stripped_input
        
        input_tokens = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )

        target_tokens = self.tokenizer(
            target_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )


        return {
            "input_ids": input_tokens["input_ids"].squeeze(),
            "attention_mask": input_tokens["attention_mask"].squeeze(),
            "labels": target_tokens["input_ids"].squeeze(),
        }

    def decode_error_correction(logits):
        #turn back into a string
        ...

    def decrypt(self, prediction): # collapse proba dists.
        if self.extra_decoder is None:
            max_indices = torch.argmax(prediction, dim=2)
            #raise Exception("fix the dimensions as we have not checked it yet, ex max_indicies is probaly not on 0")
            result_str = "".join([self.ASR_model.labels[idx] for idx in max_indices.flatten()])
            result_str = result_str.replace("-", "").replace("|", " ")
        else:
            result_str = self.extra_decoder(prediction.squeeze()).replace("|"," ")
        return result_str
    
    def __call__(self, wav_files, original_statements = None, ground_truths = None):
        output_strings_pre_error_corr = []
        output_strings_post_error_corr = []
        output_strings_only_gc = []
        with torch.no_grad():
            output_wavs = self.preproc_encoder(wav_files)
            for i,audio in enumerate(output_wavs):
                waveform_latents = self.encoder(audio)[0][-1]
                decoded_logits = self.ASR_model(waveform_latents)
                decoded_logits = decoded_logits.permute(1, 0, 2)  
                result_str = self.decrypt(decoded_logits).lower()

                input_to_corr = {"input":None, "output":"THIS IS NOT RELEVANT FOR TESTING"}
                input_to_corr["input"] = result_str.lower() #tokenizer("Hi my name is Jim")['input_ids']
                input_to_corr_only_gc = {"input":None, "output":"THIS IS NOT RELEVANT FOR TESTING"}
                input_to_corr_only_gc['input'] = original_statements[i]
                
                encoded_str = self.preproc_error_correction(input_to_corr) # Token
                encoded_str_only_gc = self.preproc_error_correction(input_to_corr_only_gc) # Token
                try: # first handle the t5 cae
                    outputs = self.error_correction_model.generate(encoded_str['input_ids'].unsqueeze(0), max_length=128, num_beams=5, early_stopping=True)
                    output_only_gc = self.error_correction_model.generate(encoded_str_only_gc['input_ids'].unsqueeze(0), max_length=128, num_beams=5, early_stopping=True)
                except:
                    outputs = self.error_correction_model(encoded_str['input_ids'].unsqueeze(0),encoded_str['attention_mask'].unsqueeze(0),encoded_str['input_ids'].unsqueeze(0)) # last input_ids is just so it computed well, it doesnt have a real effect
                    outputs = torch.argmax(outputs,-1)
                    
                    #outputs = self.error_correction_model.generate()

                    ###outputs = self.error_correction_model.generate(encoded_str['input_ids'].unsqueeze(0),encoded_str['attention_mask'].unsqueeze(0),128,encoded_str['input_ids'][0:5],tokenizer.eos_token_id)
                    #the first 10 are used for the make this sentence gramtically correct.... as a starting token
                    output_only_gc = self.error_correction_model(encoded_str_only_gc['input_ids'].unsqueeze(0),encoded_str_only_gc['attention_mask'].unsqueeze(0),encoded_str_only_gc['input_ids'].unsqueeze(0)) # last input_ids is just so it computed well, it doesnt have a real effect
                    output_only_gc = torch.argmax(output_only_gc,-1)
                corrected_sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                output_only_gc = self.tokenizer.decode(output_only_gc[0], skip_special_tokens=True)
                
                #grammar_fixed_str = self.decode_error_correction(encoded_str)

                output_strings_pre_error_corr.append(result_str)
                output_strings_post_error_corr.append(corrected_sentence)
                output_strings_only_gc.append(output_only_gc)

        wer_results = self._compute_wer_(output_strings_pre_error_corr,output_strings_post_error_corr,ground_truths, original_statements, output_strings_only_gc)
        
        return {"pre":output_strings_pre_error_corr,"post":output_strings_post_error_corr,"gt":ground_truths, "wers" : wer_results}
    
    def _compute_wer_(self, asr_output,gc_output,ground_truth, original_statements = None,GC_only = None):
        if ground_truth is None:
            print(f"ground truth is None, please specify ground truth to use the WER feature")
            return None
        wer_ASR = []
        wer_GC = []
        wer_only_asr_gic = []
        wer_only_GC_gic_to_gcs = []

        for i in range(len(asr_output)):
            wer_ASR.append(word_error_rate(asr_output[i].capitalize(),ground_truth[i])) # This should never be zero since the ground truth is not in the spoken sentence
            wer_GC.append(word_error_rate(gc_output[i].capitalize(),ground_truth[i]))
            wer_only_asr_gic.append(word_error_rate(asr_output[i].capitalize(),original_statements[i]))
            wer_only_GC_gic_to_gcs.append(word_error_rate(GC_only[i].capitalize(),ground_truth[i]))

        return {"wer_ASR":wer_ASR,"wer_GC":wer_GC, "wer_ASR_GIC" : wer_only_asr_gic, "wer_GC_only" : wer_only_GC_gic_to_gcs}
    # ASR to GCS
    # ASR -> GC -> GCS
    # ASR -> GIS
    # GIS -> GC -> GCS


if __name__ == "__main__":
    ## LOAD ASR
    info = torch.load("/home/m33murra/test_vocal/26_model",map_location="cpu")
    asr = simple_transformer_v3()
    asr.load_state_dict(info["state_dict"])
    asr.eval()


    ## From Literature ASR ASR
    #asr = simple_converter_from_literature
    #asr.cpu()
    #asr.eval()

    ## LOAD ERROR CORR
    tokenizer = T5Tokenizer.from_pretrained("/home/m33murra/test_vocal/error_correction/t5_fine_tuned_optuna")
    #loaded_model = T5ForConditionalGeneration.from_pretrained("/home/m33murra/test_vocal/error_correction/t5_fine_tuned_optuna")
    model_info = torch.load("/home/m33murra/test_vocal/error_correction/best_model_transformer.torch")
    hyp = model_info['hyp']
    loaded_model = TransFormerModel(tokenizer.vocab_size,hyp['d_model'],hyp["nhead"],hyp['nlayers'],hyp['nlayers'],hyp['dim_feedforward'],128,tokenizer.pad_token_id)
    loaded_model.base_model = None # Just to differentiate btw this and the fine-tuned version later on
    loaded_model.load_state_dict(model_info['state_dict'])
    loaded_model.cpu()
    loaded_model.eval()
    
    # Additional decoder if desired (doesnt really have an effect on the output)
    greedyDecoder = GreedyCTCDecoder(labels=asr.labels)
    model_pipeline = Pipeline(asr.ƒ,asr,loaded_model,tokenizer=tokenizer,extra_decoder=greedyDecoder)
    wav_files = ['/home/m33murra/test_vocal/wavs/I think that I deser.wav',
                 '/home/m33murra/test_vocal/wavs/First of all public.wav',
                 "/home/m33murra/test_vocal/wavs/I have seen your adv.wav",
                 "/home/m33murra/test_vocal/wavs/Michael went to a tr.wav",
                 "/home/m33murra/test_vocal/wavs/Moved by my curiosit.wav",
                 "wavs/I think you don t kn.wav",
                 "wavs/Sitting in any table.wav",
                 "/home/m33murra/test_vocal/wavs/Also when it s winte.wav",
                 "/home/m33murra/test_vocal/wavs/For languages Chines.wav",
                 "/home/m33murra/test_vocal/wavs/For the education ev.wav",
                 "/home/m33murra/test_vocal/wavs/I really enjoy soap.wav",
                 "/home/m33murra/test_vocal/wavs/Once on the car he s.wav",                
                 "/home/m33murra/test_vocal/wavs/Secondly social fact.wav",
                 "/home/m33murra/test_vocal/wavs/China s education is.wav",
                 "/home/m33murra/test_vocal/wavs/Through heat transfe.wav",
                 "/home/m33murra/test_vocal/wavs/The production cost.wav",
                 "/home/m33murra/test_vocal/wavs/Then how does refrig.wav",
                 "/home/m33murra/test_vocal/wavs/This would lead to p.wav"]

    original_statements = [
        "I think that I deserve to have a job here because with my swimming and climbing knowledge I can entertain the children",
        "First of all, public transport provide the experience of the journey",
        "I have seen your advertisement and I am more than delighted to take part in a summer camp as an assistant",
        "Michael went to a trip to Poland by himself",
        "Moved by my curiosity i dug up and found a paper which contained a map",
        "I think you don't know our national team",
        "Sitting in any table you have a wonderful view of the workshop-like kitchen where you can see the chefs working, so you can see how they make the food that you will eat",
        "Also, when it's winter, you can see something called the northern lights",
        "For languages, Chinese and English are the main languages which using in Hong Kong",
        "For the education, every developed country is concerning with the issue",
        "I really enjoy soap opera I have seen a lot of them, but in my opinion de best one is an English one called fawlty Towers",
        "Once on the car, he starts to drive like a maniac",
        "Secondly, social factors guided the growth of planes",
        "China's education is typical exam-oriented education",
        "Through heat transfer from tube to the water, bath water is heated",
        "The production cost of solar energy is higher than that of fossil fuel energy, which therefore impedes its popularization",
        "Then how does refrigerator come into being",
        "This would lead to price inflation and could cause an economic slowdown as affordability would be a concern"

    ]

    ground_truths = [
        "I think that I deserve to have a job here because, with my swimming and climbing knowledge, I can entertain the children.",
        "First of all, public transport provides the experience of the journey.",
        "I have seen your advertisement, and I would be more than delighted to take part in a summer camp as an assistant.",
        "Michael went on a trip to Poland by himself.",
        "Moved by my curiosity, I dug up and found a piece of paper that contained a map.",
        "I don't think you know our national team.",
        "Sitting at any table, you have a wonderful view of the workshop-like kitchen, where you can see the chefs working, so you can see how they make the food that you will eat.",
        "Also, when it's winter, you can see something called the northern lights.",
        "For languages, Chinese and English are the main languages which are used in Hong Kong.",
        "For education, every developed country is concerned with the issue.",
        "I really enjoy soap operas. I have seen a lot of them, but in my opinion, the best one is an English one called Fawlty Towers.",
        "Once in the car, he starts to drive like a maniac.",
        "Secondly, social factors have guided the growth of planes.",
        "China's education is a typical exam-oriented education.",
        "Through heat transfer from the tube to the water, the bath water is heated.",
        "The production cost of solar energy is higher than that of fossil fuel energy, which therefore impedes its popularity.",
        "Then how does the refrigerator come into being?",
        "This would lead to price inflation and could cause an economic slowdown as affordability would become a concern."
        ]    

    out = model_pipeline(wav_files,original_statements, ground_truths)
    for i in range(len(out['pre'])):
        print("PRE:----")
        print(out['pre'][i])
        print("POST:---")
        print(out['post'][i])
     
     #return {"wer_ASR":wer_ASR,"wer_GC":wer_GC, "wer_ASR_GIC" : wer_only_asr_gic, "wer_GC_only" : wer_only_GC_gic_to_gcs}
    print(np.round(100*np.mean([x for x in out['wers']['wer_ASR_GIC']]),4)) # WER entire pipeline 
    print(np.round(100*np.mean([x for x in out['wers']['wer_ASR']]),4)) # WER from ASR to GIC

    print(np.round(100*np.mean([x for x in out['wers']['wer_GC_only']]),4)) # WER GC from GIC to GCS
    print(np.round(100*np.mean([x for x in out['wers']['wer_GC']]),4)) # Wer from ASR to GT GCS (expect to be bad)

    # ASR to GCS
    # ASR -> GC -> GCS
    # ASR -> GIS
    # GIS -> GC -> GCS

    print("---")
    print(np.round(100*np.std([x for x in out['wers']['wer_ASR_GIC']]),4))
    print(np.round(100*np.std([x for x in out['wers']['wer_ASR']]),4))
    print(np.round(100*np.std([x for x in out['wers']['wer_GC_only']]),4))
    print(np.round(100*np.std([x for x in out['wers']['wer_GC']]),4))

    print("")

    print(len(out['pre']))

=