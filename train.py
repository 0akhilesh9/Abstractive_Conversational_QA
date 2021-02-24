import gc
import time
import copy
import torch
import GPUtil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils as utils
import basemodel as basemodel

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def train_model():
    history_length = int(utils.config.get('train', 'historyLength'))
    input_data = utils.CoQADataset(utils.config.get('train', 'trainFile'), history_length)
    train_loader = DataLoader(input_data, batch_size=1,shuffle=True)
    model = basemodel.BaseModel(384, 64, 30, use_gpu = eval(utils.config.get('train', 'useGPU')))

    CUDA = torch.cuda.is_available()
    if CUDA and eval(utils.config.get('train', 'useGPU')):
        model = model.cuda()
    model.train()
    start = time.time()
    batch_size = int(utils.config.get('train', 'batchSize'))
    lr = float(utils.config.get('train', 'learningRate'))
    beta1 = float(utils.config.get('train', 'beta1'))
    beta2 = float(utils.config.get('train', 'beta2'))
    l2_weight_decay = float(utils.config.get('train', 'l2WeightDecay'))
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=l2_weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=lr,  momentum=0.9)
    criterion = nn.NLLLoss()
    
    for step, input_batch in enumerate(train_loader):
       gc.collect()
       print("Step " + str(step))
       para_tokens = [x[0] for x in input_batch["para_tokens"]]
       batch_data = []
       for q_a in input_batch["question_answer_list"]:
           if q_a["invalid_flag"]:
               continue
           q_a["question"] = [q_a["question"][0], [x[0] for x in q_a["question"][1]]]
           for i in range(len(q_a["history_questions"])):
               q_a["history_questions"][i] = [q_a["history_questions"][i][0], [x[0] for x in q_a["history_questions"][i][1]]]
           for i in range(len(q_a["history_answers_input"])):
               q_a["history_answers_input"][i] = [q_a["history_answers_input"][i][0], [x[0] for x in q_a["history_answers_input"][i][1]]]
           for i in range(len(q_a["history_answers_span"])):
               q_a["history_answers_span"][i] = [q_a["history_answers_span"][i][0], [x[0] for x in q_a["history_answers_span"][i][1]]]
           q_a["para_tokens"] = para_tokens
       
           batch_data.append(copy.deepcopy(q_a))
       
       print("batch start - " + str(len(batch_data)))
       for i in range(0,len(batch_data)):

           if batch_size*i >= len(batch_data):
               break

           batch_input = batch_data[batch_size*i : min(batch_size*(i+1), len(batch_data))]
           optimizer.zero_grad()
           (prob_start, prob_end, prob_ans, start_pos, end_pos, ans_type) = model.forward(batch_input)
           
           if eval(utils.config.get('train', 'useGPU')):
                start_pos = torch.tensor(start_pos).cuda()
                end_pos = torch.tensor(end_pos).cuda()
                ans_type = torch.tensor(ans_type).cuda()
           else:
                start_pos = torch.tensor(start_pos)
                end_pos = torch.tensor(end_pos)
                ans_type = torch.tensor(ans_type)
                
           log_prob_start = torch.log(prob_start)
           log_prob_end = torch.log(prob_end)
           log_prob_ans = torch.log(prob_ans)
           loss_start = criterion(log_prob_start, start_pos)
           loss_end = criterion(log_prob_end, end_pos)
           loss_ans = criterion(log_prob_ans, ans_type)
           overall_loss = loss_start + loss_end + loss_ans
        
           if utils.config.get('train', 'useGPU'):
                del start_pos, end_pos, ans_type, log_prob_start, log_prob_end, log_prob_ans, loss_start, loss_end, loss_ans
                torch.cuda.empty_cache()
           
           overall_loss.backward()
           optimizer.step()
    
           del batch_input, prob_start, prob_end, prob_ans
           gc.collect()
           torch.cuda.empty_cache()
           
       print("batch end")
       
       if (step>0) and (step % int(utils.config.get('train', 'saveModelFreq')) == 0 ) and (eval(utils.config.get('train', 'saveModel'))):
           torch.save(model.state_dict(), utils.config.get('train', 'savePath'))
           print("Model saved after %d steps"%step)
            
    if eval(utils.config.get('train', 'saveModel')):
           torch.save(model.state_dict(), utils.config.get('train', 'savePath'))
    print("Time taken: " + str(time.time()-start))