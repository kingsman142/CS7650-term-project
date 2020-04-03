import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def initialize_word_tfidf_extractor():
    return TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 3),
        max_features=25000,
        sublinear_tf=True,
        strip_accents='unicode'
    )

def initialize_char_tfidf_extractor():
    return TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 5),
        max_features=60000,
        sublinear_tf=True,
        strip_accents='unicode'
    )

def find_spearman_score(y, pred_y):
    if np.ndim(y) == 2:
        count = 0
        sum = 0
        for i in range(pred_y.shape[1]):
            corr = stats.spearmanr(y[:, i], pred_y[:, i])[0]
            if np.isnan(corr):
                continue
            count = count + 1
            sum = sum + corr
        return sum / count
    else:
        corr = stats.spearmanr(y, pred_y)[0]
    return corr


data = pd.read_csv('train.csv').fillna(' ')
#test_data = pd.read_csv('test.csv').fillna(' ')
train_data, test_data = train_test_split(data, test_size=0.07, random_state=8080)

train_question = train_data['question_body']
train_answer = train_data['answer']
train_question_title = train_data['question_title']

test_question = test_data['question_body']
test_answer = test_data['answer']
test_question_title = test_data['question_title']

column_tags = train_data.columns[-30:]

for column_tag in column_tags:
    train_data[column_tag + '_dup'] = (train_data[column_tag].values >= 0.5) * 1

question_column_tags = column_tags[:21]
answer_column_tags = column_tags[21:]

feature_extractor = initialize_word_tfidf_extractor()
feature_extractor.fit(pd.concat([train_question, test_question]))
train_question_features_word = feature_extractor.transform(train_question)
test_question_features_word = feature_extractor.transform(test_question)

feature_extractor = initialize_char_tfidf_extractor()
feature_extractor.fit(pd.concat([train_question, test_question]))
train_question_features_char = feature_extractor.transform(train_question)
test_question_features_char = feature_extractor.transform(test_question)

feature_extractor = initialize_word_tfidf_extractor()
feature_extractor.fit(pd.concat([train_answer, test_answer]))
train_answer_features_word = feature_extractor.transform(train_answer)
test_answer_features_word = feature_extractor.transform(test_answer)

feature_extractor = initialize_char_tfidf_extractor()
feature_extractor.fit(pd.concat([train_answer, test_answer]))
train_answer_features_char = feature_extractor.transform(train_answer)
test_answer_features_char = feature_extractor.transform(test_answer)

feature_extractor = initialize_word_tfidf_extractor()
feature_extractor.fit(pd.concat([train_question_title, test_question_title]))
train_question_title_features_word = feature_extractor.transform(train_question)
test_question_title_features_word = feature_extractor.transform(test_question)

feature_extractor = initialize_char_tfidf_extractor()
feature_extractor.fit(pd.concat([train_question_title, test_question_title]))
train_question_title_features_char = feature_extractor.transform(train_question)
test_question_title_features_char = feature_extractor.transform(test_question)

train_features_q = hstack([train_question_features_word,
                           train_question_features_char,
                           train_question_title_features_word,
                           train_question_title_features_char])
train_features_a = hstack([train_answer_features_word,
                           train_answer_features_char])

test_features_q = hstack([test_question_features_word,
                          test_question_features_char,
                          test_question_title_features_word,
                          test_question_title_features_char])
test_features_a = hstack([test_answer_features_word,
                          test_answer_features_char])

train_features_q = train_features_q.tocsr()
train_features_a = train_features_a.tocsr()

test_features_q = test_features_q.tocsr()
test_features_a = test_features_a.tocsr()

#submission = pd.DataFrame.from_dict({'qa_id': test_data['qa_id']})
spearman_scores_train = []
spearman_scores_test = []

for question_column_tag in question_column_tags:
    print(question_column_tag)
    Y = train_data[question_column_tag + '_dup']
    model = LogisticRegression(C=0.4)
    model.fit(train_features_q, Y)
    pred_y = model.predict_proba(train_features_q)[:, 1]
    spearman_score_train = find_spearman_score(train_data[question_column_tag], pred_y)
    spearman_scores_train.append(spearman_score_train)
    print("Train:")
    print(spearman_score_train)
    pred_y = model.predict_proba(test_features_q)[:, 1]
    spearman_score_test = find_spearman_score(test_data[question_column_tag], pred_y)
    if not np.isnan(spearman_score_test):
        spearman_scores_test.append(spearman_score_test)
    print("Test:")
    print(spearman_score_test)


    #submission[question_column_tag] = model.predict_proba(test_features_q)[:, 1]

for answer_column_tag in answer_column_tags:
    print(answer_column_tag)
    Y = train_data[answer_column_tag + '_dup']
    model = LogisticRegression(C=0.4)
    model.fit(train_features_a, Y)
    pred_y = model.predict_proba(train_features_a)[:, 1]
    spearman_score_train = find_spearman_score(train_data[answer_column_tag], pred_y)
    spearman_scores_train.append(spearman_score_train)
    print("Train:")
    print(spearman_score_train)
    pred_y = model.predict_proba(test_features_a)[:, 1]
    spearman_score_test = find_spearman_score(test_data[answer_column_tag], pred_y)
    if not np.isnan(spearman_score_test):
        spearman_scores_test.append(spearman_score_test)
    print("Test:")
    print(spearman_score_test)


    #submission[answer_column_tag] = model.predict_proba(test_features_a)[:, 1]

print('Training score:')
print(np.mean(spearman_scores_train))
print('Testing score:')
print(spearman_scores_test)
print(np.mean(spearman_scores_test))

#submission.to_csv('submission.csv', index=False)
#print(submission.head())