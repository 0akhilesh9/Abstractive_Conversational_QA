import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


class BaseModel(torch.nn.Module):
    def __init__(self, max_seq_length, max_q_length, max_a_length, embedding_dim=768, prev_history=2, use_gpu=False, bert_model=None):
        super(BaseModel, self).__init__()

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        if bert_model != None:
            self.bert_model.load_state_dict(bert_model)
        
        if True:
            for name, param in self.bert_model.named_parameters():
                if "encoder.layer.11" not in name:
                    param.requires_grad = False
        
        self.max_seq_length = max_seq_length
        self.max_q_length = max_q_length
        self.max_a_length = max_a_length
        self.embedding_dim = embedding_dim
        self.prev_history = prev_history
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bi_gru_layer1 = torch.nn.GRU((prev_history*2+1)*embedding_dim, embedding_dim, batch_first=True, bidirectional=True)
        self.linear_start = torch.nn.Linear((prev_history*2+3)*embedding_dim, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.bi_gru_layer2 = torch.nn.GRU(2*embedding_dim, embedding_dim, batch_first=True, bidirectional=True)
        self.linear_end = torch.nn.Linear((prev_history*2+3)*embedding_dim, 1)
        self.answer_type_layer = torch.nn.Linear((prev_history*2+3)*embedding_dim, 3)
        self.CUDA = torch.cuda.is_available() and use_gpu
        
#    from pytorch_memlab import profile, set_target_gpu
#    @profile        
    def get_representation(self, document_spans, sent_tokens, para_tokens, sent_length, document_stride=128):
        # para_ids = self.tokenizer.convert_tokens_to_ids(para_tokens)
        sent_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        out_repr = []
        for i in range(len(para_tokens)):
            out_repr.extend([torch.zeros(768)])
    
    
        for pointer_loc, length in document_spans:
            # tmp_para_ids = para_ids[pointer_loc: pointer_loc + length] + self.tokenizer.convert_tokens_to_ids(["[SEP]"])
            tmp_para_ids = self.tokenizer.convert_tokens_to_ids(para_tokens[pointer_loc: pointer_loc + length]) + self.tokenizer.convert_tokens_to_ids(["[SEP]"])
            indexed_tokens = sent_ids + tmp_para_ids
            segments_ids = [0] * len(sent_ids) + [1] * len(tmp_para_ids)
            attention_mask = [1] * len(indexed_tokens)
            
            while len(indexed_tokens) < self.max_seq_length:
                indexed_tokens.append(0)
                attention_mask.append(0)
                segments_ids.append(0)

            assert len(indexed_tokens) == self.max_seq_length
            assert len(attention_mask) == self.max_seq_length
            assert len(segments_ids) == self.max_seq_length

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            attention_mask_tensor = torch.tensor([attention_mask])
        
            # if False and self.CUDA:
            if self.CUDA:
                tokens_tensor = tokens_tensor.cuda()
                segments_tensors = segments_tensors.cuda()
                attention_mask_tensor = attention_mask_tensor.cuda()
            
            
            encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors, attention_mask = attention_mask_tensor)
            #with torch.no_grad():
                #encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors, attention_mask = attention_mask_tensor)
            
            # tmp = BertModel.from_pretrained('bert-base-uncased')
            # encoded_layers, _ = tmp(tokens_tensor, segments_tensors, attention_mask = attention_mask_tensor)
            
            token_embeddings = torch.stack(encoded_layers, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)    # [# tokens, # layers, # features]
            
            token_vec = []
            for token in token_embeddings:
                token_vec.append(token[-1])     # final hidden state of the last layer
            
            token_vec = token_vec[len(sent_ids):len(sent_ids)+len(tmp_para_ids)-1] # Selecting just the embeddings of the paragraph
            token_vec = token_vec[:length]
            
            for i in range(length):
                if out_repr[i+pointer_loc].sum() == 0:
                    out_repr[i+pointer_loc] = token_vec[i]
                else:
                    out_repr[i+pointer_loc] = out_repr[i+pointer_loc].add(token_vec[i]) / 2
                    
            del tmp_para_ids, indexed_tokens, tokens_tensor, segments_tensors, attention_mask_tensor, encoded_layers, _, token_embeddings, token_vec
            torch.cuda.empty_cache()
            
        del sent_ids, sent_tokens, para_tokens
        torch.cuda.empty_cache()
        
        return torch.stack(out_repr)

    def forward(self, input_batch): #question_text, history_questions, history_answers_input, space_delim_para_tokens, invalid_flag, start_token_position, end_token_position):
        batch_questions = []
        start_pos = []
        end_pos = []
        ans_type = []
        for question_details in input_batch:
            start_pos.append(question_details["start_token_position"])
            end_pos.append(question_details["end_token_position"])
            if(question_details["input_answer_text"] == "yes"):
                ans_type.append(1)
            elif (question_details["input_answer_text"] == "no"):
                ans_type.append(2)
            else:
                ans_type.append(0)
            embedding_list = []
            token_vec = self.get_representation(question_details["question"][0], question_details["question"][1], question_details['para_tokens'], self.max_q_length)
            embedding_list.append(token_vec)

            for history_question in question_details["history_questions"]:
                # history_question is a tuple "(question,)" -- DataLoader converts list of history questions to list of tuples --- [(hq1,),(hq2,),(hq3,),....]
                token_vec = self.get_representation(history_question[0], history_question[1], question_details['para_tokens'], self.max_q_length)
                embedding_list.append(token_vec)

            for history_answer in question_details["history_answers_input"]:
                # history_question is a tuple "(question,)" -- DataLoader converts list of history questions to list of tuples --- [(hq1,),(hq2,),(hq3,),....]
                token_vec = self.get_representation(history_answer[0], history_answer[1], question_details['para_tokens'], self.max_q_length)
                embedding_list.append(token_vec)

            embedding_list = torch.cat(embedding_list, 1)
            missing_history = self.prev_history - len(question_details["history_questions"])
            if missing_history > 0:
                # For each missing question and answer
                num_of_paragraph_tokens = len(question_details['para_tokens'])

                if self.CUDA:
                    embedding_list = torch.cat([embedding_list.cuda(), torch.cuda.FloatTensor(num_of_paragraph_tokens, missing_history*2*self.embedding_dim).fill_(0)], 1)
                else:
                    embedding_list = torch.cat([embedding_list, torch.FloatTensor(num_of_paragraph_tokens, missing_history*2*self.embedding_dim).fill_(0)], 1)

            batch_questions.append(embedding_list)
            
        if self.CUDA:
            batch_inp = torch.stack(batch_questions).cuda()
        else:
            batch_inp = torch.stack(batch_questions)
            
        # single example is converted to batch of one element
        M_1, _ = self.bi_gru_layer1(batch_inp)
        concat_M_1 = torch.cat((batch_inp, M_1), 2)
        linear_out_start = self.linear_start(concat_M_1)
        linear_out_start = torch.squeeze(linear_out_start, 2)
        prob_start = self.softmax(linear_out_start)
        M_2, _ = self.bi_gru_layer2(M_1)
        concat_M_2 = torch.cat((batch_inp, M_2), 2)
        linear_out_end = self.linear_end(concat_M_2)
        linear_out_end = torch.squeeze(linear_out_end, 2)
        prob_end = self.softmax(linear_out_end)
        linear_out_answer = self.answer_type_layer(concat_M_2)

        if self.CUDA:
            end_span_ind = torch.unsqueeze(torch.tensor(end_pos).cuda(), 1)
        else:
            end_span_ind = torch.unsqueeze(torch.tensor(end_pos), 1)
            
        ind_for_gather = end_span_ind.repeat(1, linear_out_answer.shape[2]).view(linear_out_answer.shape[0], 1, linear_out_answer.shape[2])
        answer_type_scores = torch.gather(linear_out_answer, 1, ind_for_gather)
        answer_type_scores = torch.squeeze(answer_type_scores, 1)
        prob_ans = self.softmax(answer_type_scores)
        
        if self.CUDA:
            del end_span_ind, ind_for_gather, answer_type_scores, concat_M_1, M_1, concat_M_2, M_2, linear_out_end, linear_out_start
            torch.cuda.empty_cache()
        
        return (prob_start, prob_end, prob_ans, start_pos, end_pos, ans_type)