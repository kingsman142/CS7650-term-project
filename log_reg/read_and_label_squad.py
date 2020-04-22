import os
import pickle

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def initialize_word_tfidf_extractor():
    return TfidfVectorizer(
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
        strip_accents='unicode'
    )


def initialize_char_tfidf_extractor():
    return TfidfVectorizer(
        analyzer='char',
        ngram_range=(1, 4),
        max_features=40000,
        sublinear_tf=True,
        strip_accents='unicode'
    )


if __name__ == "__main__":

    '''
    Using context for now, since we won't have enough features if we don't
    
    if len(sys.argv) != 2:
        print("Usage: python read_and_label_squad.py [use_context]")
        sys.exit(0)
    if sys.argv[1] is not 'True' and sys.argv[1] is not 'False':
        print("use_context must be True or False")
        sys.exit(0)

    if sys.argv[1] is 'True':
        use_context = True
    else:
        use_context = False
    '''
    use_context = True

    if not os.path.isdir('squad_labelled'):
        os.mkdir('squad_labelled')

    original_data = pd.read_csv('../train.csv').fillna(' ')

    train_data = pd.read_csv('../squad_dataset/train-v2.0.csv').fillna(' ')
    dev_data = pd.read_csv('../squad_dataset/dev-v2.0.csv').fillna(' ')

    train_question = train_data['question']
    train_answer = train_data['answer']
    train_question_context = train_data['context']

    dev_question = dev_data['question']
    dev_answer = dev_data['answer']
    dev_question_context = dev_data['context']

    column_tags = original_data.columns[-30:]

    question_column_tags = column_tags[:21]
    answer_column_tags = column_tags[21:]

    feature_extractor = initialize_word_tfidf_extractor()
    feature_extractor.fit(pd.concat([train_question, dev_question]))
    train_question_features_word = feature_extractor.transform(train_question)
    dev_question_features_word = feature_extractor.transform(dev_question)

    feature_extractor = initialize_char_tfidf_extractor()
    feature_extractor.fit(pd.concat([train_question, dev_question]))
    train_question_features_char = feature_extractor.transform(train_question)
    dev_question_features_char = feature_extractor.transform(dev_question)

    feature_extractor = initialize_word_tfidf_extractor()
    feature_extractor.fit(pd.concat([train_answer, dev_answer]))
    train_answer_features_word = feature_extractor.transform(train_answer)
    dev_answer_features_word = feature_extractor.transform(dev_answer)

    feature_extractor = initialize_char_tfidf_extractor()
    feature_extractor.fit(pd.concat([train_answer, dev_answer]))
    train_answer_features_char = feature_extractor.transform(train_answer)
    dev_answer_features_char = feature_extractor.transform(dev_answer)

    feature_extractor = initialize_word_tfidf_extractor()
    feature_extractor.fit(pd.concat([train_question_context, dev_question_context]))
    train_question_context_features_word = feature_extractor.transform(train_question)
    dev_question_context_features_word = feature_extractor.transform(dev_question)

    feature_extractor = initialize_char_tfidf_extractor()
    feature_extractor.fit(pd.concat([train_question_context, dev_question_context]))
    train_question_context_features_char = feature_extractor.transform(train_question)
    dev_question_context_features_char = feature_extractor.transform(dev_question)

    if use_context:
        train_features_q = hstack([train_question_features_word,
                                   train_question_features_char,
                                   train_question_context_features_word,
                                   train_question_context_features_char])
        train_features_a = hstack([train_answer_features_word,
                                   train_answer_features_char])

        dev_features_q = hstack([dev_question_features_word,
                                 dev_question_features_char,
                                 dev_question_context_features_word,
                                 dev_question_context_features_char])
        dev_features_a = hstack([dev_answer_features_word,
                                 dev_answer_features_char])
    else:
        train_features_q = hstack([train_question_features_word,
                                   train_question_features_char])
        train_features_a = hstack([train_answer_features_word,
                                   train_answer_features_char])

        dev_features_q = hstack([dev_question_features_word,
                                 dev_question_features_char])
        dev_features_a = hstack([dev_answer_features_word,
                                 dev_answer_features_char])

    train_features_q = train_features_q.tocsr()
    train_features_a = train_features_a.tocsr()

    dev_features_q = dev_features_q.tocsr()
    dev_features_a = dev_features_a.tocsr()

    question_models = pickle.load(open('./saved_models/log_reg_question_models.sav', 'rb'))
    answer_models = pickle.load(open('./saved_models/log_reg_answer_models.sav', 'rb'))

    for question_column_tag in question_column_tags:
        train_data[question_column_tag] = question_models[question_column_tag].predict_proba(train_features_q)[:, 1]
        dev_data[question_column_tag] = question_models[question_column_tag].predict_proba(dev_features_q)[:, 1]

    for answer_column_tag in answer_column_tags:
        train_data[answer_column_tag] = answer_models[answer_column_tag].predict_proba(train_features_a)[:, 1]
        dev_data[answer_column_tag] = answer_models[answer_column_tag].predict_proba(dev_features_a)[:, 1]

    train_data.to_csv('./squad_labelled/train-v2.0_labeled_log_reg.csv', index=False)
    dev_data.to_csv('./squad_labelled/dev-v2.0_labeled_log_reg.csv', index=False)
