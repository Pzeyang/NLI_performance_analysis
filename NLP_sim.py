#Loading the required libraries
import pandas as pd
import numpy as np
import numpy
import csv
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import simhash
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tokenize import word_tokenize

## Quora question Pair Dataset
# Loading the training and testing data
df = pd.read_csv('Data/train.csv')
df['question1'] = df['question1'].apply(str)
df['question2'] = df['question2'].apply(str)
from sklearn.model_selection import train_test_split
seed = 123
train, test = train_test_split(df)

def gen_accuracy(y_pred, y_actual):
    """Function to calculate the accuracy of a model, returns the accuracy
        Args: 
            y_pred: predicted values
            y_actual: actual values"""
    
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_actual[i]:
            count = count+1
    return (count/len(y_pred))*100

def write_csv(result, id_col, path):
    result_final = pd.DataFrame(data = list(zip(id_col, result)), columns = ['id', 'prediction'])
    result_final.to_csv(path, index = False)
    
 #### Sim Hash ####
def ham_dist(text1, text2):
    return simhash.Simhash(text1, f = 8).distance(simhash.Simhash(text2, f = 8))

train['sim_dist'] = train.apply(lambda x: ham_dist(x['question1'], x['question2']), axis = 1)

train.groupby('is_duplicate').agg('mean')['sim_dist']

test['sim_dist'] = test.apply(lambda x: ham_dist(x['question1'], x['question2']), axis = 1)

result_df = []
for i in test['sim_dist']:
    if i > 2.5:
        result_df.append(0)
    else:
        result_df.append(1)
        
count = 0
for i in range(len(result_df)):
    if result_df[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1
        
accuracy = count/len(result_df)
print('accuracy: '+ str(accuracy))

print('f1_socre:' + str(f1_score(test['is_duplicate'], result_df, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_df, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_df, average = 'weighted')))

write_csv(result_df, test['id'], 'results/NLP/quora/simhash.csv')



###NER Similarity
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union== 0:
        return 0
    else:
        return float(intersection) / union
    
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
            else:
                    continue
    return continuous_chunk


train['ne1'] = train.apply(lambda x:get_continuous_chunks(x['question1']), axis = 1)
train['ne2'] = train.apply(lambda x: get_continuous_chunks(x['question2']), axis = 1)

train['jac_sim'] = train.apply(lambda x: jaccard_similarity(x['ne1'], x['ne2']), axis = 1)
train.groupby('is_duplicate').agg('mean')['jac_sim']

test['ne1'] = test.apply(lambda x:get_continuous_chunks(x['question1']), axis = 1)
test['ne2'] = test.apply(lambda x: get_continuous_chunks(x['question2']), axis = 1)

test['jac_sim'] = test.apply(lambda x: jaccard_similarity(x['ne1'], x['ne2']), axis = 1)

result_ner = []
for i in test['jac_sim']:
    if i > .21:
        result_ner.append(1)
    else:
        result_ner.append(0)
        
count = 0
for i in range(len(result_ner)):
    if result_ner[i] == test['is_duplicate'].to_numpy()[i]:
        count = count +1
accuracy = count/len(result_ner)
print('accuracy: ' + str(accuracy))

print('f1_socre:' + str(f1_score(test['is_duplicate'], result_ner, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_ner, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_ner, average = 'weighted')))

write_csv(result_ner, test['id'], 'results/NLP/quora/ner.csv')

#### Cosine Distance ####
#training and testing
q = list(train['question1']) + list(train['question2']) + list(test['question1']) + list(test['question2'])

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)

vectorizer.fit(q)

#train_sent1 = train['#1 String'] + " " + train['#2 String']
#test_sent = test['#1 String']+ " " + test['#2 String']

# sent_vect_train1 = vectorizer.transform(train['#1 String'])
# sent_vect_train2 = vectorizer.transform(train['#2 String'])

def cosine_sim(text1, text2):
    
    tfidf = vectorizer.transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

train['cosine_sim'] = train.apply(lambda x: cosine_sim(x['question1'], x['question2']), axis= 1)

train.groupby('is_duplicate').agg('mean')['cosine_sim']

test['cosine_sim'] = test.apply(lambda x: cosine_sim(x['question1'], x['question2']), axis= 1)

result_cosine = []
for i in test['cosine_sim'].to_numpy():
    if i> .63:
        result_cosine.append(1)
    else:
        result_cosine.append(0)
        
count = 0
for i in range(len(result_cosine)):
    if result_cosine[i] == test['is_duplicate'].to_numpy()[i]:
        count = count+1
        
accuracy = count/len(result_cosine)
print('accuracy: ' + str(accuracy))

print('f1_socre:' + str(f1_score(test['is_duplicate'], result_cosine, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_cosine, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_cosine, average = 'weighted')))

write_csv(result_cosine, test['id'], 'results/NLP/quora/cosine.csv')

#### MSR Dataset ####
# Loading the training and testing data
train = pd.read_csv(r'Data/msr_paraphrase_train.txt', sep = '\t', quoting=csv.QUOTE_NONE)
test = pd.read_csv(r'Data/msr_paraphrase_test.txt', sep = '\t', quoting=csv.QUOTE_NONE)

###simhash###
def ham_dist(text1, text2):
    return simhash.Simhash(text1, f = 8).distance(simhash.Simhash(text2, f = 8))

train['sim_dist'] = train.apply(lambda x: ham_dist(x['#1 String'], x['#2 String']), axis = 1)

train.groupby('Quality').agg('mean')['sim_dist']

test['sim_dist'] = test.apply(lambda x: ham_dist(x['#1 String'], x['#2 String']), axis = 1)

result_df = []
for i in test['sim_dist']:
    if i > 2.1:
        result_df.append(0)
    else:
        result_df.append(1)
        
count = 0
for i in range(len(result_df)):
    if result_df[i] == test['Quality'].to_numpy()[i]:
        count = count +1
        
accuracy = count/len(result_df)
print('accuracy: '+str(accuracy))

print('f1_socre:' + str(f1_score(test['Quality'], result_df, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_df, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_df, average = 'weighted')))

write_csv(result_df, test['#1 ID'], 'results/NLP/MSR/simhash.csv')

####NER similarity ####
train['ne1'] = train.apply(lambda x:get_continuous_chunks(x['#1 String']), axis = 1)
train['ne2'] = train.apply(lambda x: get_continuous_chunks(x['#2 String']), axis = 1)

train['jac_sim'] = train.apply(lambda x: jaccard_similarity(x['ne1'], x['ne2']), axis = 1)
train.groupby('Quality').agg('mean')['jac_sim']
test['ne1'] = test.apply(lambda x:get_continuous_chunks(x['#1 String']), axis = 1)
test['ne2'] = test.apply(lambda x: get_continuous_chunks(x['#2 String']), axis = 1)

test['jac_sim'] = test.apply(lambda x: jaccard_similarity(x['ne1'], x['ne2']), axis = 1)
result_ner = []
for i in test['jac_sim']:
    if i > .41:
        result_ner.append(1)
    else:
        result_ner.append(0)
        
count = 0
for i in range(len(result_ner)):
    if result_ner[i] == test['Quality'].to_numpy()[i]:
        count = count +1
        
print('accuracy: ' + str(count/len(result_ner)))
print('f1_socre:' + str(f1_score(test['Quality'], result_ner, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_ner, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_ner, average = 'weighted')))

###Cosine distance ####

#training and testing
sent = list(train['#1 String']) + list(train['#2 String']) + list(test['#1 String']) + list(test['#2 String'])

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)

vectorizer.fit(sent)

#train_sent1 = train['#1 String'] + " " + train['#2 String']
#test_sent = test['#1 String']+ " " + test['#2 String']

# sent_vect_train1 = vectorizer.transform(train['#1 String'])
# sent_vect_train2 = vectorizer.transform(train['#2 String'])


def cosine_sim(text1, text2):
    
    tfidf = vectorizer.transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

train['cosine_sim'] = train.apply(lambda x: cosine_sim(x['#1 String'], x['#2 String']), axis= 1)
train.groupby('Quality').agg('mean')['cosine_sim']
test['cosine_sim'] = test.apply(lambda x: cosine_sim(x['#1 String'], x['#2 String']), axis= 1)

result_cosine = []
for i in test['cosine_sim'].to_numpy():
    if i> .7:
        result_cosine.append(1)
    else:
        result_cosine.append(0)
count = 0
for i in range(len(result_cosine)):
    if result_cosine[i] == test['Quality'].to_numpy()[i]:
        count = count+1
        
accuracy = count/len(result_cosine)
print('accuracy: ' + str(accuracy))

print('f1_socre:' + str(f1_score(test['Quality'], result_cosine, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_cosine, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_cosine, average = 'weighted')))

write_csv(result_cosine, test['#1 ID'], 'results/NLP/MSR/cosine.csv')
write_csv(result_cosine, test['#1 ID'], 'results/NLP/MSR/cosine.csv')
