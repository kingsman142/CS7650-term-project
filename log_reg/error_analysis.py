import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if not os.path.exists("plots_error_analysis"):
    os.mkdir("plots_error_analysis")

train_data = pd.read_csv('../train.csv').fillna(' ')

train_features_q = pickle.load(open('./saved_models/log_reg_train_features_q.sav', 'rb'))
train_features_a = pickle.load(open('./saved_models/log_reg_train_features_a.sav', 'rb'))
question_models = pickle.load(open('./saved_models/log_reg_question_models.sav', 'rb'))
answer_models = pickle.load(open('./saved_models/log_reg_answer_models.sav', 'rb'))

column_tags = train_data.columns[-30:]

for column_tag in column_tags:
    train_data[column_tag + '_dup'] = (train_data[column_tag].values >= 0.5) * 1

question_column_tags = np.array(column_tags[:21])
answer_column_tags = np.array(column_tags[21:])

done = True
column_erros = []
for question_column_tag in question_column_tags:
    pred_y = question_models[question_column_tag].predict_proba(train_features_q)[:, 1]
    diff = np.absolute(train_data[question_column_tag] - pred_y)
    greater = np.where(diff >= 0.5)[0]
    if done and question_column_tag == 'question_type_choice':
        print(train_data.iloc[greater[60]]['question_title'])
        # print(train_data.iloc[0]['question_body'])
        print("Ans")
        print(train_data.iloc[greater[60]]['answer'])
        done = False
    column_erros.append(len(greater))
column_erros = np.array(column_erros)

top_5_indices = np.array(column_erros).argsort()[-5:][::-1]
plt.bar(question_column_tags[top_5_indices], (column_erros[top_5_indices]))
plt.xticks(rotation='vertical')
plt.xlabel('Question_tag')
plt.ylabel('No of training samples')
plt.tight_layout()
plt.savefig('./plots_error_analysis/question_error_analysis.png')
plt.clf()

column_erros = []
for answer_column_tag in answer_column_tags:
    pred_y = answer_models[answer_column_tag].predict_proba(train_features_a)[:, 1]
    diff = np.absolute(train_data[answer_column_tag] - pred_y)
    column_erros.append(len(np.where(diff >= 0.5)[0]))
column_erros = np.array(column_erros)

top_5_indices = np.array(column_erros).argsort()[-5:][::-1]
plt.bar(answer_column_tags[top_5_indices], (column_erros[top_5_indices]))
plt.xticks(rotation='vertical')
plt.xlabel('Answer_tag')
plt.ylabel('No of training samples')
plt.tight_layout()
plt.savefig('./plots_error_analysis/answer_error_analysis.png')
