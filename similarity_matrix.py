import torch
from tqdm import tqdm
import time
import pickle
from torch.utils.data import Dataset
#from torchvision import datasets
#from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import os
import gc
import argparse
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset
from data_loader import load_mnli,load_hans, load_snli
from torch.nn.functional import cosine_similarity
import numpy as np

input_path = './'
BATCH_SIZE = 256

class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        # print(loss)
        loss = loss.mean()
        # print(loss)
        return loss



class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn = None):
        super(MainModel,self).__init__(config)
        self.num_labels = 3
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-uncased",config = config)
        self.hidden = nn.Linear(768, 2*(self.num_labels))
        self.classifier = nn.Linear(2*(self.num_labels), self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels,device):
              
        output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        #output = output.last_hidden_state
        #output = output[:,0,:]
        return output


def inference(model, dataloader, tokenizer, device, data = 'snli'):
    model.eval()
    nb_test_steps = 0
    cnt=0
    hidden_state_outputs = np.full((13,32,768), 0, dtype=float)
    start_time = time.time()
    
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        cnt+=1
        print(cnt)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets, device = device)
        
        hidden_states = output.hidden_states
        
        
        #print(type(hidden_states))
        
        #print(f"Number of layers: {len(hidden_states)}")
        
        #for i, layer_output in enumerate(hidden_states):
        #	print(f"Layer {i}: {layer_output.shape}")
        
        num_layers = len(hidden_states)  # 13
        similarity_matrix = np.zeros((num_layers, num_layers))
        
        # Step 1: Normalize and flatten each layer
        
        normalized_layers = []
        
        for i in range(num_layers):
        	layer = hidden_states[i]  # shape: (32, 512, 768)
        	layer = layer.view(-1, layer.shape[-1])  # (32*512, 768)
        	layer = torch.nn.functional.normalize(layer, p=2, dim=1)  # L2 norm
        	normalized_layers.append(layer)  # list of (16384, 768)
        	
        # Step 2: Compute cosine similarity between all layer pairs
        for i in range(num_layers):
        	for j in range(num_layers):
        		sim = torch.sum(normalized_layers[i] * normalized_layers[j], dim=1)  # (16384,)
        		avg_sim = sim.mean().item()
        		similarity_matrix[i][j] = avg_sim
        
        
        
        #break
        
    
    np.save("similarity_matrix.npy", similarity_matrix)
    print(similarity_matrix)
    #np.save("similarity_matrix.npy", similarity_matrix)
    
def read_dataset(data_path):
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data



def main():
    gc.collect()

    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    model = MainModel.from_pretrained(args.input_model_path, loss_fn = LossFunction())
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    model.to(device)
    #print("Testing started")
    #loding HANS data



    snli_data = load_snli(file_path='./snli_1.0/snli_1.0_train.txt', tokenizer=tokenizer)

    snli_test_dataloader = DataLoader(snli_data, shuffle = False, batch_size=BATCH_SIZE)
    
    inference(model, snli_test_dataloader, tokenizer, device)
	
    
    end = time.time()
    total_time = end - start
    with open('live_bert_base_test.txt', 'a') as fh:
        fh.write(f'Total testing time : {total_time}\n')

    print(f"Total test time : {total_time}")
if __name__ == '__main__':
    main()
