import gc
import sys
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import copy
import torch.nn as nn
from torch.utils.data import Dataset
import torch.optim as optim
import GPUtil

import utils as utils
import basemodel as basemodel
from pytorch_pretrained_bert import BertTokenizer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


train_df = utils.read_json(utils.config.get('train', 'trainFile'))
print(len(train_df["data"]))
train_set, val_set = train_df["data"][:6800], train_df["data"][6800: ]
src_train = open("data/src-train.txt", "w")
tgt_train = open("data/tgt-train.txt", "w")
src_val = open("data/src-val.txt", "w")
tgt_val = open("data/tgt-val.txt", "w")

for paragraph in train_set:
    question_list = paragraph["questions"]
    answers_list = paragraph["answers"]
    for ind in range(len(question_list)):
        if answers_list[ind]["input_text"] == "yes" or answers_list[ind]["input_text"] == "no":
            continue
        question = " ".join(tokenizer.tokenize(question_list[ind]["input_text"]))
        answer =  " ".join(tokenizer.tokenize(answers_list[ind]["span_text"]))
        src_line = question + " " + answer
        tgt_line = " ".join(tokenizer.tokenize(answers_list[ind]["input_text"]))

        src_line = src_line.replace(" ##", '')
        tgt_line = tgt_line.replace(" ##", '')
        src_train.write(src_line+"\n")
        tgt_train.write(tgt_line+"\n")

for paragraph in val_set:
    question_list = paragraph["questions"]
    answers_list = paragraph["answers"]
    for ind in range(len(question_list)):
        if answers_list[ind]["input_text"] == "yes" or answers_list[ind]["input_text"] == "no":
            continue
        question = " ".join(tokenizer.tokenize(question_list[ind]["input_text"]))
        answer = " ".join(tokenizer.tokenize(answers_list[ind]["span_text"]))
        src_line = question + " " + answer
        tgt_line = " ".join(tokenizer.tokenize(answers_list[ind]["input_text"]))

        src_line = src_line.replace(" ##", '')
        tgt_line = tgt_line.replace(" ##", '')
        src_val.write(src_line+"\n")
        tgt_val.write(tgt_line+"\n")

src_train.close()
tgt_train.close()
src_val.close()
tgt_val.close()

test_df = utils.read_json(utils.config.get('train', 'devFile'))
src_test = open("data/src-test.txt", "w")
tgt_test_label = open("data/tgt-test", "w")
for paragraph in test_df["data"]:
    question_list = paragraph["questions"]
    answers_list = paragraph["answers"]
    for ind in range(len(question_list)):
        question = " ".join(tokenizer.tokenize(question_list[ind]["input_text"]))
        answer =  " ".join(tokenizer.tokenize(answers_list[ind]["span_text"]))
        src_line = question + " " + answer
        src_label_line = " ".join(tokenizer.tokenize(answers_list[ind]["input_text"]))

        src_line = src_line.replace(" ##", '')
        src_label_line = src_label_line.replace(" ##", '')
        src_test.write(src_line+"\n")
        tgt_test_label.write(src_label_line+"\n")