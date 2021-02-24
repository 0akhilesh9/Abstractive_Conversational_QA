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
import argparse

import utils as utils
import basemodel as basemodel
from eval_utils import compute_f1_score_batch
from utils import get_start_end_indices

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--write_pred_file', default=None, type=str, help='prediction file')
args = parser.parse_args()

history_length = int(utils.config.get('train', 'historyLength'))
input_data = utils.CoQADataset(utils.config.get('train', 'devFile'), history_length)
test_loader = DataLoader(input_data[0:10], batch_size=1, shuffle=False)


CUDA = torch.cuda.is_available()
model = basemodel.BaseModel(384, 64, 30, use_gpu = eval(utils.config.get('train', 'useGPU')))
state_dict = torch.load('resources/model.pth')
model.load_state_dict(state_dict)
if CUDA and eval(utils.config.get('train', 'useGPU')):
    model = model.cuda()
model.eval()
start = time.time()
batch_size = int(utils.config.get('train', 'batchSize'))

with torch.no_grad():
    for step, input_batch in enumerate(test_loader):
        gc.collect()
        print("Step " + str(step))
        para_tokens = [x[0] for x in input_batch["para_tokens"]]
        batch_data = []
        for q_a in input_batch["question_answer_list"]:
            if q_a["invalid_flag"]:
                continue
            q_a["question"] = [q_a["question"][0], [x[0] for x in q_a["question"][1]]]
            for i in range(len(q_a["history_questions"])):
                q_a["history_questions"][i] = [q_a["history_questions"][i][0],
                                               [x[0] for x in q_a["history_questions"][i][1]]]
            for i in range(len(q_a["history_answers_input"])):
                q_a["history_answers_input"][i] = [q_a["history_answers_input"][i][0],
                                                   [x[0] for x in q_a["history_answers_input"][i][1]]]
            for i in range(len(q_a["history_answers_span"])):
                q_a["history_answers_span"][i] = [q_a["history_answers_span"][i][0],
                                                  [x[0] for x in q_a["history_answers_span"][i][1]]]
            q_a["para_tokens"] = para_tokens

            batch_data.append(copy.deepcopy(q_a))

        print("batch start - " + str(len(batch_data)))
        prob_start_list = []
        prob_end_list = []
        for i in range(0, len(batch_data)):
            if batch_size * i >= len(batch_data):
                break
            batch_start = time.time()
            batch_input = batch_data[batch_size * i: min(batch_size * (i + 1), len(batch_data))]
            # print(batch_size*i, min(batch_size*(i+1), len(batch_data)))
            (prob_start, prob_end, prob_ans, start_pos, end_pos, ans_type) = model.forward(batch_input)
            prob_start_list.append(prob_start.squeeze(0).numpy())
            prob_end_list.append(prob_end.squeeze(0).numpy())
            # print(time.time() - batch_start)
            #       GPUtil.showUtilization()
            del batch_input
            #torch.cuda.empty_cache()
        result = get_start_end_indices(prob_start_list, prob_end_list)
        f1_score = compute_f1_score_batch(batch_data, result, write_pred_file=args.write_pred_file)
        print("batch end")

print("Time taken: " + str(time.time() - start))