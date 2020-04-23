import numpy as np
import pandas as pd
import re

from collections import Counter

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')

print(train.shape, test.shape)

labels = [f for f in train.columns if f not in test.columns]

train[labels] = train[labels].applymap(lambda x: 1 if x >= 0.5 else 0)

py = {l:np.mean(train[l]) for l in labels}

# Training

pwtitle_y1, pwtitle_y0 = {}, {}
pwbody_y1, pwbody_y0 = {}, {}
pwanswer_y1, pwanswer_y0 = {}, {}

d = len(set(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([train['question_title'][i].lower() 
                                                   for i in range(train.shape[0])])).split()))\
+len(set(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([train['question_body'][i].lower() 
                                                for i in range(train.shape[0])])).split()))\
+len(set(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([train['answer'][i].lower() 
                                                for i in range(train.shape[0])])).split()))

for l in labels:
    
    # y1
    tr = train[train[l] == 1].reset_index(drop=True)
    
    counts_title = Counter(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([tr['question_title'][i].lower() 
                                                                  for i in range(tr.shape[0])])).split())
    counts_body = Counter(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([tr['question_body'][i].lower() 
                                                                 for i in range(tr.shape[0])])).split())
    counts_answer = Counter(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([tr['answer'][i].lower() 
                                                                   for i in range(tr.shape[0])])).split())
    
    nrmlz1 = sum((counts_title+counts_body+counts_answer).values())
    pwtitle_y1[l] = {k:(v+1)/(nrmlz1+d) for k, v in counts_title.items()}
    pwbody_y1[l] = {k:(v+1)/(nrmlz1+d) for k, v in counts_body.items()}
    pwanswer_y1[l] = {k:(v+1)/(nrmlz1+d) for k, v in counts_answer.items()}
    
    # y0
    tr = train[train[l] == 0].reset_index(drop=True)
    
    counts_title = Counter(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([tr['question_title'][i].lower() 
                                                                  for i in range(tr.shape[0])])).split())
    counts_body = Counter(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([tr['question_body'][i].lower() 
                                                                 for i in range(tr.shape[0])])).split())
    counts_answer = Counter(re.sub(r'([^\s\w]|_)+', ' ', ' '.join([tr['answer'][i].lower() 
                                                                   for i in range(tr.shape[0])])).split())
    
    nrmlz0 = sum((counts_title+counts_body+counts_answer).values())
    pwtitle_y0[l] = {k:(v+1)/(nrmlz0+d) for k, v in counts_title.items()}
    pwbody_y0[l] = {k:(v+1)/(nrmlz0+d) for k, v in counts_body.items()}
    pwanswer_y0[l] = {k:(v+1)/(nrmlz0+d) for k, v in counts_answer.items()}

# Testing

scores = np.zeros((test.shape[0], len(labels)))

for i in range(test.shape[0]):
    
    for j in range(len(labels)):
        
        l = labels[j]
        
        # y1        
        title_score = np.sum([np.log(pwtitle_y1[l][w]) if w in pwtitle_y1[l] else 1/(nrmlz1+d)
                              for w in re.sub(r'([^\s\w]|_)+', ' ', 
                                              test['question_title'][i].lower()).split()])
        body_score = np.sum([np.log(pwbody_y1[l][w]) if w in pwbody_y1[l] else 1/(nrmlz1+d) 
                             for w in re.sub(r'([^\s\w]|_)+', ' ',
                                             test['question_body'][i].lower()).split()])
        answer_score = np.sum([np.log(pwanswer_y1[l][w]) if w in pwanswer_y1[l] else 1/(nrmlz1+d) 
                               for w in re.sub(r'([^\s\w]|_)+', ' ', 
                                               test['answer'][i].lower()).split()])
        score1 = np.log(py[l])+title_score+body_score+answer_score
        
        # y0        
        title_score = np.sum([np.log(pwtitle_y0[l][w]) if w in pwtitle_y0[l] else 1/(nrmlz0+d)
                              for w in re.sub(r'([^\s\w]|_)+', ' ', 
                                              test['question_title'][i].lower()).split()])
        body_score = np.sum([np.log(pwbody_y0[l][w]) if w in pwbody_y0[l] else 1/(nrmlz0+d) 
                             for w in re.sub(r'([^\s\w]|_)+', ' ',
                                             test['question_body'][i].lower()).split()])
        answer_score = np.sum([np.log(pwanswer_y0[l][w]) if w in pwanswer_y0[l] else 1/(nrmlz0+d) 
                               for w in re.sub(r'([^\s\w]|_)+', ' ', 
                                               test['answer'][i].lower()).split()])
        score0 = np.log(1-py[l])+title_score+body_score+answer_score
        
        # wrap-up
        if score1-score0 > 10:
            scores[i, j] = 1
        elif score1-score0 < -10:
            scores[i, j] = 0
        else:
            scores[i, j] = 1/(1+np.exp(score0-score1))

# Submission to Kaggle

scores = scores+np.random.random()/10000 # avoids error in kaggle

submission = pd.concat([pd.DataFrame(data=test['qa_id'], columns=['qa_id']),
                        pd.DataFrame(data=scores, columns=labels)], 
                       axis=1)
print(submission.shape)
submission.head()

submission.to_csv('submission.csv', index=False)


