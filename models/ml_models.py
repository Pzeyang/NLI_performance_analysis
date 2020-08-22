#import statements

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

class ml_models():
    
    def __init__(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain
            
    def logistic_regression(self, 
                            xtest,
                            solver = 'saga',
                            penalty = 'l2',
                            max_iter = 1000,
                            random_state = 0,
                            C = 10):
 
        lr = LogisticRegression(solver = solver, 
                                penalty = penalty,
                                random_state=random_state,
                                max_iter = max_iter,
                                C = C)
        lr.fit(self.xtrain, self.ytrain)
        return lr.predict(xtest)
    
    def svm(self,
            xtest,
            kernel= 'rbf', 
            C = 100):
        svc = SVC(kernel = kernel, 
                  C = C)
        svc.fit(self.xtrain, self.ytrain)
        return svc.predict(xtest)
    
    def random_forest(self, 
                      xtest,
                      n_estimators = 600,
                      bootstrap = True,
                      max_features = 'sqrt',
                      max_depth = 200):
       
        random_forest = RandomForestClassifier(n_estimators=n_estimators, 
                               bootstrap = bootstrap,
                               max_features = max_features)
        # Fit on training data
        random_forest.fit(self.xtrain, self.ytrain)
        return random_forest.predict(xtest)
    
    def xgbclassifier(self, xtest, 
                      learning_rate=.01,
                      max_depth = 3,
                      n_estimators = 300):
        model = XGBClassifier(max_depth=max_depth,
                              learning_rate=learning_rate,
                              n_estimators=n_estimators)                      
        model.fit(self.xtrain, self.ytrain)
        return model.predict(xtest)
    
    def adaboost(self,
                 xtest,
                 learning_rate = .1,
                 n_estimators = 100):
        model = AdaBoostClassifier(learning_rate = learning_rate,
                                   n_estimators = n_estimators)
        model.fit(self.xtrain, self.ytrain)
        return model.predict(xtest)
  
    def voting_classifier(self, ytest):
        
        xgb = XGBClassifier(max_depth=3,learning_rate=0.01,n_estimators=312)                      
        
        ada_boost = AdaBoostClassifier()
        
        random_forest = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
        lr = LogisticRegression(solver = 'saga', penalty = 'elasticnet', l1_ratio=.5, random_state=0)
        
        rf = random_forest = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
        
        svm= SVC(probability=True)

        
        model = VotingClassifier(estimators=[('logistic', lr), 
                                             ('ada', ada_boost),
                                             ('random_forest', rf),
                                             ('xgb', xgb),
                                             ('svm', svm)], 
                       voting='soft', weights=[1,1,3,1,1]).fit(self.xtrain,self.ytrain)
        return model.predict(ytest)