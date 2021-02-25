1) Source code for the Baseline model was not available and hence, this was implemented completely from scratch.
OpenNmt library was used for PgNet abstractive generation - https://github.com/OpenNMT/OpenNMT-py/tree/c199de0fb738c33c02c76a392005e94f6b89add3
URL for BERT Abstractive generation - https://colab.research.google.com/drive/1MxKZGtQ9SSBjTK5ArsZ5LKhkztzg52RV

2) For BERT Abstractive generation following files were created by referring to the above collab resource
	a) bert_gen.py - abstract_text

3) Commands

a) Train Extractive model - python test.py

b) predict Extractive model - python predict.py

Ex - python predict.py --write_pred_file Results/Pred_GPUmodel.txt --model saved_models/resources_GPUmodel.pth --test_size 50

c) BERT abstractive layer prediction - please see below

d) BLEU calculation - python calc_blue.py



4) pytorch-1.3.1, pytorch_pretrained_bert-0.6.2, ijson - 2.5.1, configparser-4.0.2, pandas0.25.3, GPUtil-1.4.0, matplotlib - 3.1.1, numpy1.17.2, nltk - 3.4.5, scipy-1.3.3


Miscelleneous:

basemodel.py file contains the code for the base QA model
train.py file contains the code for training the base QA model
test.py file is the main file from where the execution (training and BERT abstractive generation) starts
bert_gen.py file contains the code for abstract text generation
utils.py contains utilities like configuration, reading files, writing files, and other formatting functions that are used across other files
eval_utils.py for evaluation purpose
gen_pgnet_data.py for abstract text using pg net
predict.py contains code for making predictions
project.config contains the configuration for the whole application



For abstractive text generation using BERT, the prediction file should be of the following format:

Q: question_text
	
A: span_answer_text

Q: question_text
	
A: span_answer_text
.
.
.


and the actual answer text file should have the format:

answer1

answer2
.
.
.

The ordering of these two files should be the same.

Command to run: 
python test.py --abstract --pred_file resources/pred.txt --actual_file resources/act.txt

This will create a CSV with all the calculated metric values and also saves the plots.
Use '--show_graphs' flag to see the plots during execution itself.
