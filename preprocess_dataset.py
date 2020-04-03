import os
import sys
import json
import random
import pandas as pd
import numpy as np

# useful constants
TRAIN_PATH = os.path.join("train.csv")
TEST_PATH = os.path.join("test.csv")
USEFUL_COLUMN_NAMES = ["question_body", "answer"]
VALIDATION_SIZE = 500

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
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

# find the column names that pertain to the scores
column_names = train_data.columns
column_names_for_scores = column_names[11:]
print(column_names_for_scores)

# preprocess scores
print("Preprocessing scores...")
train_scores = train_data[column_names_for_scores].values
new_train_scores = []
for row in train_scores:
    new_scores = row.tolist()
    new_train_scores.append(new_scores)

# extract only the useful column names for text preprocessing
train_data = train_data[USEFUL_COLUMN_NAMES]
test_data = test_data[USEFUL_COLUMN_NAMES]

train_questions = []
train_answers = []
test_questions = []
test_answers = []

# preprocess question and answer text
word2idx = {"[NULL]": 0}
idx_counter = 1
longest_sentence = 0
print("Preprocessing training text...")
for index, row in train_data.iterrows(): # iterate over training data
    # read in question and answer on this row
    question, answer = row["question_body"], row["answer"]

    # convert from "This is my question?" to ["This", "is", "my", "question", "[EOQ]"]
    question_tokenized = preprocess_text(question)
    answer_tokenized = preprocess_text(answer)
    longest_sentence = max(longest_sentence, len(question_tokenized))
    longest_sentence = max(longest_sentence, len(answer_tokenized))

    # prepare some variables so we can convert ["This", "is", "my", "question", "[EOQ]"] to [23, 486, 3, 54, 128]
    question_numbered = []
    answer_numbered = []

    for word in question_tokenized:
        if word not in word2idx:
            word2idx[word] = idx_counter
            idx_counter += 1
        question_numbered.append(word2idx[word])
    for word in answer_tokenized:
        if word not in word2idx:
            word2idx[word] = idx_counter
            idx_counter += 1
        answer_numbered.append(word2idx[word])
    question_numbered += [0] * (8172 - len(question_numbered))
    answer_numbered += [0] * (8172 - len(answer_numbered))

    train_questions.append(question_numbered)
    train_answers.append(answer_numbered)
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
        if word not in word2idx:
            word2idx[word] = idx_counter
            idx_counter += 1
        question_numbered.append(word2idx[word])
    for word in answer_tokenized:
        if word not in word2idx:
            word2idx[word] = idx_counter
            idx_counter += 1
        answer_numbered.append(word2idx[word])
    question_numbered += [0] * (8172 - len(question_numbered))
    answer_numbered += [0] * (8172 - len(answer_numbered))

    test_questions.append(question_numbered)
    test_answers.append(answer_numbered)

# concatenate question/answer text and scores
print("Concatenating three columns together...")
new_train_data = []
new_test_data = []
for index, row in enumerate(new_train_scores):
    new_row = [train_questions[index], train_answers[index], row]
    new_train_data.append(new_row)
for index, row in enumerate(test_questions):
    new_row = [test_questions[index], test_answers[index]]
    new_test_data.append(new_row)

# shuffle training data so we can create a validation set
random.shuffle(new_train_data)
new_val_data = new_train_data[0:VALIDATION_SIZE]
new_train_data = new_train_data[VALIDATION_SIZE:]

# turn training and test data into DataFrames
new_train_data = pd.DataFrame(new_train_data, columns = ["question", "answer", "scores"])
new_val_data = pd.DataFrame(new_val_data, columns = ["question", "answer", "scores"])
new_test_data = pd.DataFrame(new_test_data, columns = ["question", "answer"])
print(new_train_data.shape, new_val_data.shape, new_test_data.shape)

# save new data
print("Saving data...")
new_train_data.to_csv(os.path.join("train_preprocessed.csv"))
new_val_data.to_csv(os.path.join("val_preprocessed.csv"))
new_test_data.to_csv(os.path.join("test_preprocessed.csv"))
with open("word2idx.json", "w+") as word2idx_file:
    json.dump(word2idx, word2idx_file)
print("\nVocabulary size: {}".format(len(word2idx)))
print("Longest sentence length: {}".format(longest_sentence))
