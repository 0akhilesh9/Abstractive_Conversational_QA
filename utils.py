import configparser
import pandas as pd
from statistics import mean 
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from nltk.translate.bleu_score import sentence_bleu
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM

import bert_gen as bert_gen


config = configparser.ConfigParser()
config.read_file(open(r"project.config"))


def read_json(file_path, encoding='utf-8'):
    contents_df = pd.read_json(file_path, encoding=encoding)
    return contents_df


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


class CoQADataset(Dataset):
  def __init__(self, filename, history_length, train_flag=True):
    self.train_flag = train_flag
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df = read_json(filename)
    self.train_data = []
    
    def check_whitespace(char_):
      if char_ == " " or char_ == "\t" or char_ == "\r" or char_ == "\n" or ord(char_) == 0x202F:
        return True
      return False
    
    for row in train_df["data"]:
      p_text = row["story"]
      sample_data = []
      
      # Split the input paragraph into tokens using space delimiter
      space_delim_para_tokens = []
      char_index_to_token_index_map = []
      last_char_space_flag = True
      for char_val in p_text:
        if check_whitespace(char_val):
          last_char_space_flag = True
        else:
          if last_char_space_flag:
            space_delim_para_tokens.append(char_val)
          else:
            space_delim_para_tokens[-1] += char_val
          last_char_space_flag = False
        char_index_to_token_index_map.append(len(space_delim_para_tokens) - 1)
      
      bert_token_to_space_token_index = []
      space_token_to_bert_token_index = []
      para_tokens = []
      for (i, token) in enumerate(space_delim_para_tokens):
        space_token_to_bert_token_index.append(len(para_tokens))
        sub_tokens = self.tokenizer.tokenize(token)
        for sub_token in sub_tokens:
          bert_token_to_space_token_index.append(i)
          para_tokens.append(sub_token)
        
      for index, question in enumerate(row["questions"]):
        question_text = question["input_text"]
        turn_id = question["turn_id"]
        start_token_position = None
        end_token_position = None
        input_answer_text = None
        span_answer_text = None
        invalid_flag = True if row['answers'][index]['input_text'] == 'unknown' else False
        
        # Convert answer span values from character based index to space delimited token based index
        if not invalid_flag:
          input_answer_text = row["answers"][index]["input_text"]
          span_answer_text = row["answers"][index]["span_text"]
          answer_offset = row["answers"][index]["span_start"]
          answer_length = len(span_answer_text)
          start_token_position = char_index_to_token_index_map[answer_offset]
          end_token_position = char_index_to_token_index_map[answer_offset + answer_length - 1]                     
          # actual_text = " ".join(doc_tokens[start_token_position:(end_token_position + 1)])
          # cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(input_answer_text))
        else:
          start_token_position = -1
          end_token_position = -1
          input_answer_text = ""
          span_answer_text = ""
        
        # Convert answer span from space delimited token based index to BERT token based index
        bert_token_start_position, bert_token_end_position = self.update_answer_span(
                para_tokens, space_delim_para_tokens, start_token_position, 
                end_token_position, input_answer_text, space_token_to_bert_token_index, 
                invalid_flag, self.train_flag)
        
        # Add history questions and answers
        history_questions = []
        history_answers_input = []
        history_answers_span = []
        for j in range(index - 1, index-history_length - 1, -1):
          if j >= 0:
            document_spans, sent_tokens = self.process_sentence_tokens(row["questions"][j]["input_text"], para_tokens, int(config.get("train","maxQueryLength")), 
                                                                  True, int(config.get("train","documentStride")), self.train_flag)
            history_questions.append([document_spans, sent_tokens])
            document_spans, sent_tokens = self.process_sentence_tokens(row["answers"][j]["span_text"], para_tokens, int(config.get("train","maxAnsLength")), 
                                                                  True, int(config.get("train","documentStride")), self.train_flag)
            history_answers_span.append([document_spans, sent_tokens])
            document_spans, sent_tokens = self.process_sentence_tokens(row["answers"][j]["input_text"], para_tokens, int(config.get("train","maxAnsLength")), 
                                                                  True, int(config.get("train","documentStride")), self.train_flag)
            history_answers_input.append([document_spans, sent_tokens])   
        
        document_spans, sent_tokens = self.process_sentence_tokens(question_text, para_tokens, int(config.get("train","maxQueryLength")), 
                                                              True, int(config.get("train","documentStride")), self.train_flag)
        question = [document_spans, sent_tokens]
        

        sample_dict = {"turn_id": turn_id,
                            "question": question,
                            "history_questions": history_questions,
                            "history_answers_input": history_answers_input,
                            "history_answers_span": history_answers_span,
                            "start_token_position": bert_token_start_position,
                            "end_token_position": bert_token_end_position,
                            "input_answer_text": input_answer_text,
                            "span_answer_text": span_answer_text,
                            "invalid_flag": invalid_flag}
          
        sample_data.append(sample_dict)
    
      self.train_data.append({"para_tokens": para_tokens, 
                        "question_answer_list": sample_data})


  def update_answer_span(self, para_tokens, space_delim_para_tokens, start_token_position, end_token_position, actual_answer_text, 
                         space_token_to_bert_token_index, invalid_flag, train_flag):
    bert_token_start_position = None
    bert_token_end_position = None
    
    if train_flag and invalid_flag:
      bert_token_start_position = -1
      bert_token_end_position = -1
    if train_flag and not invalid_flag:
      bert_token_start_position = space_token_to_bert_token_index[start_token_position]
      if end_token_position < len(space_delim_para_tokens) - 1:
        bert_token_end_position = space_token_to_bert_token_index[end_token_position + 1] - 1
      else:
        bert_token_end_position = len(para_tokens) - 1
      (bert_token_start_position, bert_token_end_position) = improve_answer_span(
          para_tokens, bert_token_start_position, bert_token_end_position, self.tokenizer, actual_answer_text)
      
    return bert_token_start_position, bert_token_end_position
    

  def process_sentence_tokens(self, sent_text, para_tokens, sent_length, invalid_flag, document_stride, train_flag):
      
    sent_tokens = self.tokenizer.tokenize(sent_text)
    sent_tokens = ["[CLS]"] + sent_tokens + ["[SEP]"]
    
    if len(sent_tokens) > sent_length:
        sent_tokens = sent_tokens[:sent_length]
        # sent_tokens = sent_tokens[-sent_length]
             
    max_para_tokens = int(config.get("train","maxSeqLength")) - len(sent_tokens) - 3
    
    document_spans = []
    pointer_loc = 0
    while pointer_loc < len(para_tokens):
      length = len(para_tokens) - pointer_loc
      if length > max_para_tokens:
        length = max_para_tokens
      document_spans.append([pointer_loc, length])
      if pointer_loc + length == len(para_tokens):
        break
      pointer_loc += min(length, document_stride)
    
    return document_spans, sent_tokens


  def __len__(self):
    return len(self.train_data)


  def __getitem__(self, idx):
    return self.train_data[idx]


def check(start, end):
  maxprod = -1
  Xpos = Ypos = 1
  assert len(start) == len(end)
  for xpos in range(len(start)):
    for ypos in range(xpos, len(end)):
      if start[xpos] * end[ypos] > maxprod:
        maxprod = start[xpos] * end[ypos]
        Xpos, Ypos = xpos, ypos
  return maxprod, Xpos, Ypos

def get_max_prob(start, end):
  maxend = [-1] * len(end)
  maxval = (-1, -1)
  assert len(start)==len(end)
  for i in range(len(end)-1, -1, -1):
    if end[i] > maxval[0]:
      maxval = end[i], i
    maxend[i] = maxval

  maxprod = -1
  maxob = ()
  for i in range(len(end)):
    if start[i] * maxend[i][0] > maxprod:
      maxprod = start[i] * maxend[i][0]
      maxob = maxprod, i, maxend[i][1]
  return maxob

def get_start_end_indices(batch_start, batch_end):
  res = []
  for i in range(len(batch_start)):
    maxprob, st, end = get_max_prob(batch_start[i], batch_end[i])
    res.append([st,end])
  return res
  
def get_bleu(gen_text, part_b):
    part_b_toks = [x for x in part_b.split(" ") if len(x)>0]
    gen_text_toks = [x for x in gen_text.split(" ") if len(x)>0]
    score = sentence_bleu([part_b_toks], gen_text_toks,  weights=(1, 0, 0, 0))
    return score

def compute_f1_score(prediction, ground_truth):
    common = Counter(prediction.split()) & Counter(ground_truth.split())
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction.split())
    recall = 1.0 * num_same / len(ground_truth.split())
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def generate_abstract_text(pred_file, actual_file, show_graphs, batch_size=5):
    pred_list = []
    act_list = []

    with open(pred_file,"r",encoding="utf-8") as fp:
       for cnt, line in enumerate(fp):
           if "A:" in line[:3]:
               pred_list.append(line[2:].strip())

    with open(actual_file,"r",encoding="utf-8") as fp:
       for cnt, line in enumerate(fp):
           act_list.append(line.strip())

    pred_file = pred_file[1130:]
    act_list = act_list[1130:]

    columns=["seq_bs","seq_f1","seq_r","seq_p","p_bs_avg","p_bs_max","p_f1_avg","p_f1_max","p_r_avg","p_r_max","p_p_avg","p_p_max",
                              "ps_bs_avg","ps_bs_max","ps_f1_avg","ps_f1_max","ps_r_avg","ps_r_max","ps_p_avg","ps_p_max"]
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    abstract_model = bert_gen.BertGen(model, batch_size)
    res = pd.DataFrame(columns = columns)
    for i in range(len(pred_list)):
        print(i)
        part_a = pred_list[i]
        part_b = act_list[i]
        
        if len(part_b) <2 and len(part_a)<2:
            continue
        
        abstract_dict = abstract_model.get_abstract_text(part_a)
        seq_text = abstract_dict["seq"]
        par_seq_gen_text = abstract_dict["par_seq"]
        par_gen_text = abstract_dict["par"]
        
        ps_f1 = []
        ps_bs = []
        ps_r = []
        ps_p = []
        
        pp_f1 = []
        pp_bs = []
        pp_r = []
        pp_p = []    
        
        s_p,s_r,s_f = compute_f1_score(seq_text, part_b)
        s_bs = get_bleu(seq_text, part_b)
        for i in range(batch_size):
            p,r,f = compute_f1_score(par_seq_gen_text[i], part_b)
            bs = get_bleu(par_seq_gen_text[i], part_b)
            ps_f1.append(f)
            ps_r.append(r)
            ps_p.append(p)
            ps_bs.append(bs)
            
            p,r,f = compute_f1_score(par_gen_text[i], part_b)
            bs = get_bleu(par_gen_text[i], part_b)
            pp_f1.append(f)
            pp_r.append(r)
            pp_p.append(p)
            pp_bs.append(bs)
            
        res=res.append(pd.DataFrame([[s_bs,s_f,s_r,s_p,
                       mean(pp_bs), max(pp_bs), mean(pp_f1), max(pp_f1), mean(pp_r), max(pp_r),mean(pp_p), max(pp_r),
                       mean(ps_bs), max(ps_bs),mean(ps_f1), max(ps_f1),mean(ps_r), max(ps_r),mean(ps_p), max(ps_p)]], columns=columns))
      
    res.to_csv("out.csv")
    
    plt.plot(range(res['p_p_max'].shape[0]), res['p_p_max'], label='parallel_precision_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['seq_p'], label='sequence_precision')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_p_max'], label='parallel_sequnce_precision_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_p_avg'], label='parallel_sequnce_precision_avg')
    plt.plot(range(res['p_p_max'].shape[0]), res['p_p_avg'], label='parallel_precision_avg')
    plt.ylabel('Precision score')
    plt.legend()
    plt.savefig('Precision.png')
    if show_graphs:
        plt.show()
    
    plt.plot(range(res['p_p_max'].shape[0]), res['p_r_max'], label='parallel_recall_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['seq_r'], label='sequence_recall')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_r_max'], label='parallel_sequnce_recall_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_r_avg'], label='parallel_sequnce_recall_avg')
    plt.plot(range(res['p_p_max'].shape[0]), res['p_r_avg'], label='parallel_recall_avg')
    plt.ylabel('Recall score')
    plt.legend()
    plt.savefig('Recall.png')
    if show_graphs:
        plt.show()
    
    plt.plot(range(res['p_p_max'].shape[0]), res['p_f1_max'], label='parallel_f1_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['seq_f1'], label='sequence_f1')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_f1_max'], label='parallel_sequnce_f1_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_f1_avg'], label='parallel_sequnce_f1_avg')
    plt.plot(range(res['p_p_max'].shape[0]), res['p_f1_avg'], label='parallel_f1_avg')
    plt.ylabel('F1 score')
    plt.legend()
    plt.savefig('F1.png')
    if show_graphs:
        plt.show()
    
    plt.plot(range(res['p_p_max'].shape[0]), res['p_bs_max'], label='parallel_blue_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['seq_bs'], label='sequence_blue')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_bs_max'], label='parallel_sequnce_blue_max')
    plt.plot(range(res['p_p_max'].shape[0]), res['ps_bs_avg'], label='parallel_sequnce_blue_avg')
    plt.plot(range(res['p_p_max'].shape[0]), res['p_bs_avg'], label='parallel_blue_avg')
    plt.ylabel('BLEU score')
    plt.legend()
    plt.savefig('BLEU.png')
    if show_graphs:
        plt.show()
    
    return res