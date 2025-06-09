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

input_path = './'
BATCH_SIZE = 32

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
        output = output.last_hidden_state
        output = output[:,0,:]
        hidden_output = self.hidden(output)
        classifier_out = self.classifier(hidden_output)
        main_prob = F.softmax(classifier_out, dim = 1)
        main_gold_prob = torch.gather(main_prob, 1, labels)
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main,main_prob


def inference(model, dataloader, tokenizer, device, data = 'snli'):
    model.eval()
    prob_lst = []
    pred_lst = []
    test_loss = 0
    bias_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    hans_dict = {1 : 'non-entailment', 2 : 'entailment'}
    mnli_dict = {0 : 'contradiction', 1 : 'neutral', 2 : 'entailment'}
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        with torch.no_grad():
            loss_main,main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets, device = device)
        test_loss += loss_main.item()
        nb_test_steps += 1
        prob_lst.extend(main_prob)
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        if data == 'hans':
            predicted_labels = torch.where((predicted_labels == 0) | (predicted_labels == 1), torch.ones_like(predicted_labels), predicted_labels)
            pred = [hans_dict[label.item()] for label in predicted_labels]
        else:
            pred = [mnli_dict[label.item()] for label in predicted_labels]
        pred_lst.extend(pred)
        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        test_accuracy += tmp_test_accuracy
        
    test_accuracy = test_accuracy / nb_test_steps
    
    return test_accuracy, pred_lst, prob_lst


def generate_prediction_file(pred_lst, output_file):
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')

def generate_prob_file(prob_lst, prob_output_file_path):
    with open(prob_output_file_path, 'w') as fh:
        for probs in prob_lst:
            for prob in probs:
                fh.write(f'{prob} ')
            fh.write('\n')


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
    print("Testing started")
    #loding HANS data



    snli_data = load_snli(file_path='./snli_1.0/snli_1.0_test.txt', tokenizer=tokenizer)

    snli_test_dataloader = DataLoader(snli_data, shuffle = False, batch_size=BATCH_SIZE)

    snli_test_accuracy, snli_pred_lst, snli_prob_lst = inference(model, snli_test_dataloader, tokenizer, device)
    print(f'\tSNLI set test accuracy: {snli_test_accuracy}')
    

    
    generate_prediction_file(snli_pred_lst, './Predictions/SNLI/pred_snli_bert_base.txt')

    generate_prob_file(snli_prob_lst, './Predictions/SNLI/prob_snli_bert_base.txt')

    end = time.time()
    total_time = end - start
    with open('live_bert_base_test.txt', 'a') as fh:
        fh.write(f'Total testing time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()
