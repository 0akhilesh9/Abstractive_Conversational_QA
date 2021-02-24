from collections import Counter
import re
import string
import numpy as np

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.history = []
        self.last = None
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.last = self.mean()
        self.history.append(self.last)
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        if self.count == 0:
            return 0.
        return self.sum / self.count

def compute_f1_score(prediction, ground_truth):
    common = Counter(prediction.split()) & Counter(ground_truth.split())
    num_same = sum(common.values())
    #print(common, num_same)
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction.split())
    recall = 1.0 * num_same / len(ground_truth.split())
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def compute_f1_score_batch(inputs, start_end):
    precision_list = []
    recall_list = []
    for inp,st_end in zip(inputs, start_end):
        actual_text = inp['span_answer_text']
        start_token_position = inp['start_token_position']
        end_token_position = inp['end_token_position']
        para_tokens = inp['para_tokens']
        predicted_text = para_tokens[st_end[0]: st_end[1]]#[start_token_position:end_token_position]
        predicted_text = ' '.join(predicted_text)
        predicted_text = predicted_text.replace(' ##', '')
        # normalize_text(predicted_text, actual_text[0])
        precision, recall, f1 = compute_f1_score(predicted_text, actual_text[0].lower())
        precision_list.append(precision)
        recall_list.append(recall)
    print('avg precision={:.2f} recall={:2f}'.format(np.mean(precision_list), np.mean(recall_list)))
    return 0 if not (np.mean(precision_list) + np.mean(recall_list)) else  (2 * np.mean(precision_list) * np.mean(recall_list)) / (np.mean(precision_list) + np.mean(recall_list))
    #return f1