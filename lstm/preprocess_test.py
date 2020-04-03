import os
import sys
import json
import random
import pandas as pd
import numpy as np

# useful constants
TEST_PATH = os.path.join("../input/google-quest-challenge/test.csv")
USEFUL_COLUMN_NAMES = ["question_body", "answer"]

print(os.listdir("../"))

def preprocess_text(text):
    # TODO: maybe convert all text to lowercase? see if that affects accuracy?
    text = text.replace(".", " [EOS] ")
    text = text.replace("?", " [EOQ] ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.split(" ")
    return text

# load data
print("Loading data...")
test_data = pd.read_csv(TEST_PATH)
with open("../input/mydata/word2idx.json", "r") as word2idx_file:
    word2idx = json.loads(word2idx_file.read())
    vocab_size = len(word2idx)

# extract only the useful column names for text preprocessing
test_data = test_data[USEFUL_COLUMN_NAMES]
test_questions = []
test_answers = []

# preprocess question and answer text
print("Preprocessing testing text...")
for index, row in test_data.iterrows(): # iterate over testing data
    # read in question and answer on this row
    question, answer = row

    # convert from "This is my question?" to ["This", "is", "my", "question", "[EOQ]"]
    question_tokenized = preprocess_text(question)
    answer_tokenized = preprocess_text(answer)

    # prepare some variables so we can convert ["This", "is", "my", "question", "[EOQ]"] to [23, 486, 3, 54, 128]
    question_numbered = []
    answer_numbered = []

    # convert ["This", "is", "my", "question", "[EOQ]"] to [23, 486, 3, 54, 128]
    for word in question_tokenized:
        question_numbered.append(word2idx[word] if word in word2idx else len(word2idx))
    for word in answer_tokenized:
        answer_numbered.append(word2idx[word] if word in word2idx else len(word2idx))
    question_numbered += [0] * (8172 - len(question_numbered))
    answer_numbered += [0] * (8172 - len(answer_numbered))

    test_questions.append(question_numbered)
    test_answers.append(answer_numbered)

# concatenate question/answer text and scores
print("Concatenating three columns together...")
new_test_data = []
for index, row in enumerate(test_questions):
    new_row = [test_questions[index], test_answers[index]]
    new_test_data.append(new_row)

# turn training and test data into DataFrames
new_test_data = pd.DataFrame(new_test_data, columns = ["question", "answer"])
print(new_test_data.shape)

# save new data
print("Saving data...")
new_test_data.to_csv(os.path.join("../test_preprocessed.csv"))
print("\nVocabulary size: {}".format(len(word2idx)))
