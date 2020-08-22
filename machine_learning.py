## Machine learning models###

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
import models.ml_models as ml_models
from sklearn.metrics import precision_score, recall_score, f1_score

### quora Data set###
#Loading the data
df = pd.read_csv('Data/train.csv')
df['question1'] = df['question1'].apply(str)
df['question2'] = df['question2'].apply(str)
df.dropna(inplace = True)
#df = df[:30000]

#training and testing
from sklearn.model_selection import train_test_split
seed = 123
train, test = train_test_split(df)
q = list(train['question1']) + list(train['question2']) + list(test['question1']) + list(test['question2'])

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

#training and testing
sent = list(train['question1']) + list(train['question2']) + list(test['question1']) + list(test['question2'])

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)

vectorizer.fit(sent)

train_sent = train['question2'] + " " + train['question1']
test_sent = test['question1']+ " " + test['question2']

sent_vect_train = vectorizer.transform(train_sent)
sent_vect_test = vectorizer.transform(test_sent)


## ML class##
ml = ml_models.ml_models(sent_vect_train, train['is_duplicate']) 

## Logistic regression###
result_lr = ml.logistic_regression(sent_vect_test)
gen_accuracy(result_lr, test['is_duplicate'].to_numpy())
print('f1_socre:' + str(f1_score(test['is_duplicate'], result_lr, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_lr, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_lr, average = 'weighted')))

write_csv(result_lr,test['id'].to_numpy(),'results/machine_learning/quora/lr.csv')

### SVM ###
#SVC
#** Uncomment to run SVM **
# result_svc = ml.svm(sent_vect_test)
# gen_accuracy(result_svc, test['is_duplicate'])
# print('f1_socre:' + str(f1_score(test['is_duplicate'], result_svc, average = 'weighted')))
# print('recall_socre:' + str(recall_score(test['is_duplicate'], result_svc, average = 'weighted')))
# print('precision_socre:' + str(precision_score(test['is_duplicate'], result_svc, average = 'weighted')))

#write_csv(result_svc,test['id'].to_numpy(),'results/machine_learning/quora/svc.csv')

### Random Forest ##
#RF
result_rf = ml.random_forest(sent_vect_test, max_depth=100, n_estimators=400)
gen_accuracy(result_rf, test['is_duplicate'].to_numpy())
print('f1_socre:' + str(f1_score(test['is_duplicate'], result_rf, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_rf, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_rf, average = 'weighted')))

write_csv(result_rf, test['is_duplicate'].to_numpy(), 'results/machine_learning/quora/random_forest.csv')

###XGBoost###
result_xg = ml.xgbclassifier(sent_vect_test, learning_rate = 1)
gen_accuracy(result_xg, test['is_duplicate'].to_numpy())
print('f1_socre:' + str(f1_score(test['is_duplicate'], result_xg, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_xg, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_xg, average = 'weighted')))
write_csv(result_xg, test['is_duplicate'].to_numpy(), 'results/machine_learning/quora/xgboost.csv')

###Adaboost
result_ada = ml.adaboost(sent_vect_test, n_estimators=400, learning_rate=1)
gen_accuracy(result_ada, test['is_duplicate'].to_numpy())
print('f1_socre:' + str(f1_score(test['is_duplicate'], result_ada, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_ada, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_ada, average = 'weighted')))

write_csv(result_ada, test['is_duplicate'].to_numpy(), 'results/machine_learning/quora/adaboost.csv')

### Voting Classifier

result_voting = ml.voting_classifier(sent_vect_test, svm = False)
gen_accuracy(result_voting, test['is_duplicate'].to_numpy())
print('f1_socre:' + str(f1_score(test['is_duplicate'], result_voting, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['is_duplicate'], result_voting, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['is_duplicate'], result_voting, average = 'weighted')))
write_csv(result_voting, test['is_duplicate'].to_numpy(), 'results/machine_learning/quora/voting.csv')


### MSR Paraphrase Data ####
# Loading the training and testing data
import csv
train = pd.read_csv(r'Data/msr_paraphrase_train.txt', sep = '\t', quoting=csv.QUOTE_NONE)
test = pd.read_csv(r'Data/msr_paraphrase_test.txt', sep = '\t', quoting=csv.QUOTE_NONE)

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

train_sent = train['#1 String'] + " " + train['#2 String']
test_sent = test['#1 String']+ " " + test['#2 String']

sent_vect_train = vectorizer.transform(train_sent)
sent_vect_test = vectorizer.transform(test_sent)

ml = ml_models.ml_models(sent_vect_train, train['Quality']) 

### Logistic regression ###
result_lr = ml.logistic_regression(sent_vect_test)
print(gen_accuracy(result_lr, test['Quality']))
print('f1_socre:' + str(f1_score(test['Quality'], result_lr, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_lr, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_lr, average = 'weighted')))

write_csv(result_lr, test['#1 ID'], 'results/machine_learning/MSR/lr.csv')

###SVM###
#SVC
result_svc = ml.svm(sent_vect_test)
gen_accuracy(result_svc, test['Quality'])
print('f1_socre:' + str(f1_score(test['Quality'], result_svc, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_svc, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_svc, average = 'weighted')))
write_csv(result_svc, test['#1 ID'], 'results/machine_learning/MSR/svm.csv')

###Random Fores####RF
result_rf = ml.random_forest(sent_vect_test)
print(gen_accuracy(result_rf, test['Quality']))
print('f1_socre:' + str(f1_score(test['Quality'], result_rf, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_rf, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_rf, average = 'weighted')))
write_csv(result_rf, test['#1 ID'], 'results/machine_learning/MSR/rf.csv')

###XG Boost
result_xg = ml.xgbclassifier(sent_vect_test)
gen_accuracy(result_xg, test['Quality'])
print('f1_socre:' + str(f1_score(test['Quality'], result_xg, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_xg, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_xg, average = 'weighted')))
write_csv(result_xg, test['#1 ID'], 'results/machine_learning/MSR/xgboost.csv')

###### Adaboost
result_ada = ml.adaboost(sent_vect_test)
gen_accuracy(result_ada, test['Quality'])
print('f1_socre:' + str(f1_score(test['Quality'], result_ada, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_ada, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_ada, average = 'weighted')))

write_csv(result_ada, test['#1 ID'], 'results/machine_learning/MSR/adaboost.csv')

### Voting Classifier
result_voting = ml.voting_classifier(sent_vect_test)
gen_accuracy(result_voting, test['Quality'])
print('f1_socre:' + str(f1_score(test['Quality'], result_voting, average = 'weighted')))
print('recall_socre:' + str(recall_score(test['Quality'], result_voting, average = 'weighted')))
print('precision_socre:' + str(precision_score(test['Quality'], result_voting, average = 'weighted')))

write_csv(result_voting, test['#1 ID'], 'results/machine_learning/MSR/voting_classifier.csv')
