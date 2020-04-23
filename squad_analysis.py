import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

def extract_question_answer(sample):
    remove_arrows = sample.split(" -> ")
    question = remove_arrows[1].replace("\"", "").replace(", Answers", "")
    answers = remove_arrows[2].split("}")[0].replace("{", "").replace("\"", "").split(", ")
    return question, answers

# NOTE: SQuAD1.1 dataset downloaded as a JSON from https://datarepository.wolframcloud.com/resources/SQuAD-v1.1
dataset = pd.read_csv(os.path.join("data", "SQuAD-v1.1.csv"), encoding = 'latin1')["QuestionAnswerSets"]

print("Number of samples: {}".format(len(dataset)))

# the dataset is formatted oddly, so extract the question and answers from the plaintext
questions = []
answers = []
for index, sample in enumerate(dataset):
    question, answer = extract_question_answer(sample)
    questions.append(question)
    answers.append(answer)

# question analysis
question_avg_length = 0
for question in questions:
    question_avg_length += len(question.split(" "))
question_avg_length /= len(questions)

# answer analysis
answer_avg_length = 0
answer_avg = 0
for answer_list in answers:
    answer_avg += len(answer_list) # number of answers given for this one question
    for answer in answer_list:
        answer_avg_length += len(answer.split(" "))
answer_avg_length /= answer_avg
answer_avg /= len(answers)

print("Average question length: {}".format(question_avg_length))
print("Average answer length: {}".format(answer_avg_length))
print("Average number of answers: {}".format(answer_avg))
