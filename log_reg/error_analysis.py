import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

train_data = pd.read_csv('../train.csv').fillna(' ')

train_features_q = pickle.load(open('./saved_models/log_reg_train_features_q.sav', 'rb'))
train_features_a = pickle.load(open('./saved_models/log_reg_train_features_a.sav', 'rb'))
question_models = pickle.load(open('./saved_models/log_reg_question_models.sav', 'rb'))
answer_models = pickle.load(open('./saved_models/log_reg_answer_models.sav', 'rb'))

column_tags = train_data.columns[-30:]

for column_tag in column_tags:
    train_data[column_tag + '_dup'] = (train_data[column_tag].values >= 0.5) * 1

question_column_tags = column_tags[:21]
answer_column_tags = column_tags[21:]

for question_column_tag in question_column_tags:
    print(question_column_tag)
    print("Number of errors where difference >= 0.5")
    pred_y = question_models[question_column_tag].predict_proba(train_features_q)[:, 1]
    diff = np.absolute(train_data[question_column_tag] - pred_y)

for answer_column_tag in answer_column_tags:
    print(answer_column_tag)
    print("Number of errors where difference >= 0.5")
    pred_y = question_models[answer_column_tag].predict_proba(train_features_a)[:, 1]
    diff = np.absolute(train_data[answer_column_tag] - pred_y)