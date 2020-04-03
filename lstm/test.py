import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from dataset import QUESTDataset
from model import QUESTModel

HIDDEN_SIZE = 100
LSTM_LAYERS = 1
EMBEDDING_DIM = 50
WORD2IDX_PATH = os.path.join("word2idx.json")

QUESTION_OUTPUT_SIZE = 21 # 21 features/attributes corresponding to the questions
ANSWER_OUTPUT_SIZE = 9 # 9 features/attributes corresponding to the answers
COLUMN_NAMES = ['question_asker_intent_understanding', 'question_body_critical',
       'question_conversational', 'question_expect_short_answer',
       'question_fact_seeking', 'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible', 'answer_relevance',
       'answer_satisfaction', 'answer_type_instructions',
       'answer_type_procedure', 'answer_type_reason_explanation',
       'answer_well_written']

saved_model_names = glob.glob("models/*")
MODEL_SAVE_DIR = max(saved_model_names, key = os.path.getctime) # get the model that was saved most recently

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load in word2idx and determine vocab size
with open(WORD2IDX_PATH, "r") as word2idx_file:
    word2idx = json.loads(word2idx_file.read())
    vocab_size = len(word2idx)

# init dataset and loaders for test dataset
test_data = QUESTDataset(data_path = os.path.join("test_preprocessed.csv"))
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)

# init training modules, such as the model, Adam optimizer and loss function
model = QUESTModel(vocab_size = vocab_size, embedding_dim = EMBEDDING_DIM, hidden_size = HIDDEN_SIZE, lstm_layers = LSTM_LAYERS, num_question_output = QUESTION_OUTPUT_SIZE, num_answer_output = ANSWER_OUTPUT_SIZE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_DIR))

# create a 'models' directory to save models
if not os.path.exists("models"):
    print("No pretrained models exist. Please run train.py first...")


# run the model on the testing dataset
print("Testing model...")
test_preds = []
for batch_id, samples in enumerate(test_loader):
    question, answer = samples["question"].to(DEVICE), samples["answer"].to(DEVICE)
    pred_scores = model(question.to(DEVICE), answer.to(DEVICE))
    test_preds.append(pred_scores[0].cpu().detach().numpy())

    if batch_id % 50 == 0:
        print("Done with {}/{} test samples...".format(batch_id+1, len(test_loader)))

# save the predictions to a file
print("Saving predictions...")
test_preds = pd.DataFrame(test_preds, columns = COLUMN_NAMES)
test_preds.to_csv("test_submission.csv")
