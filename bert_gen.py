import re
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from pytorch_pretrained_bert import BertTokenizer


class BertGen():
    def __init__(self, model, batch_size):
        self.model = model
        self.MASK = "[MASK]"
        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def masking_sequential(self, text_tokens):
        segment_tensor = torch.tensor([0] * len(text_tokens))
        for step_n in range(len(text_tokens)):
            token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
            token_id_tensor = torch.tensor([token_ids])
            predictions = self.model(token_id_tensor, segment_tensor)
            prob_distribution = Categorical(logits=F.log_softmax(predictions[0], dim=-1))
            predicted_indexes = prob_distribution.sample().tolist()
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indexes)
            text_tokens[step_n] = predicted_tokens[step_n]
        return text_tokens

    def pad_inp_text(self, text_tokens, max_seq_len):
        batch_data = [text_tokens + [self.MASK] * max_seq_len + [self.SEP] for _ in range(self.batch_size)]
        return [self.tokenizer.convert_tokens_to_ids(sentence) for sentence in batch_data]

    def masking_parallel_sequential(self, text_tokens, max_seq_length=15, max_iterations=30):
        token_count = len(text_tokens)
        batch_data = self.pad_inp_text(text_tokens, max_seq_length)
    
        for itr in range(max_iterations):
            rand_num = np.random.randint(0, max_seq_length)
            for i in range(self.batch_size):
                batch_data[i][token_count + rand_num] = self.tokenizer.convert_tokens_to_ids([self.MASK])[0]
            batch_tensor = torch.tensor(batch_data)
            batch_predictions = self.model(batch_tensor)
            
            bert_predictions = batch_predictions[:, token_count+rand_num]
            prob_distribution = torch.distributions.categorical.Categorical(logits=bert_predictions)
            predicted_batch_tokens = prob_distribution.sample().squeeze(-1)
            
            for i in range(self.batch_size):
                batch_data[i][token_count+rand_num] = predicted_batch_tokens[i].item()
        
        return [self.tokenizer.convert_ids_to_tokens(sentence) for sentence in batch_data]

    def masking_parallel(self, text_tokens, max_seq_length=15, max_iterations=30):
        token_count = len(text_tokens)
        batch_data = self.pad_inp_text(text_tokens, max_seq_length)
        
        for itr in range(max_iterations):
            batch_tensor = torch.tensor(batch_data)
            batch_predictions = self.model(batch_tensor)
            for i in range(max_seq_length):
                bert_predictions = batch_predictions[:, token_count+i]
                prob_distribution = torch.distributions.categorical.Categorical(logits=bert_predictions)
                predicted_batch_tokens = prob_distribution.sample().squeeze(-1)
                
                for j in range(self.batch_size):
                    batch_data[j][token_count+i] = predicted_batch_tokens[j].item()
                
        return [self.tokenizer.convert_ids_to_tokens(sentence) for sentence in batch_data]
    
    def get_abstract_text(self, text, max_iterations = 5):
        tokenized_txt = self.tokenizer.tokenize(text)
        tokenized_txt = tokenized_txt[:505]  # Bert limit
        max_seq_length = len(tokenized_txt) + 5
        pred_parallel_seq = self.masking_parallel_sequential(tokenized_txt, max_seq_length=max_seq_length, max_iterations=max_iterations)
        pred_parallel = self.masking_parallel(tokenized_txt, max_seq_length=max_seq_length, max_iterations=max_iterations)
        seq_text = self.masking_sequential(tokenized_txt)
        seq_text = " ".join(seq_text)
        seq_text = seq_text.replace("[MASK]","")
        seq_text = seq_text.replace("[SEP]","")
        seq_text = re.sub('[^A-Za-z0-9 ]+', '', seq_text)
        seq_text = re.sub('[ ]+', ' ', seq_text)
        output = {"par_seq":[], "par":[], "seq": seq_text}
    
        for i in range(self.batch_size):
            pred = " ".join(pred_parallel_seq[i])
            pred = pred.replace("[MASK]","")
            pred = pred.replace("[SEP]","")
            pred = re.sub('[^A-Za-z0-9 ]+', '', pred)
            pred = re.sub('[ ]+', ' ', pred)
            output["par_seq"].append(pred)
            
            pred = " ".join(pred_parallel[i])
            pred = pred.replace("[MASK]","")
            pred = pred.replace("[SEP]","")
            pred = re.sub('[^A-Za-z0-9 ]+', '', pred)
            pred = re.sub('[ ]+', ' ', pred)
            output["par"].append(pred)

        return output