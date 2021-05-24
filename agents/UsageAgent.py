#! /usr/bin/env python3

# Imports 
# plotting 
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import sklearn.metrics

# More ML Models
import sklearn as sk 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tqdm import tqdm
#from scripts.helper_functions import Helper
 

class Usage_Agent:
    def __init__(self, input_df, device):
        self.input = input_df
        self.device = device


    def train_test_split(self, df, date, train_start='2013-11-01'):
      select_vars =  [self.device + '_usage', self.device+ '_usage_lag_1', self.device+ '_usage_lag_2',	'active_last_2_days']
      df = df[select_vars]
      X_train = df.loc[train_start:date, df.columns != self.device + '_usage']
      y_train = df.loc[train_start:date, df.columns == self.device + '_usage']
      X_test  = df.loc[date, df.columns != self.device + '_usage']
      y_test  = df.loc[date , df.columns == self.device + '_usage']
      return X_train, y_train, X_test, y_test

#################### MINE ###################################################################################################
    ################Mine##############################
    ## PIPELINE FUNCTIONS ONLY HAVE TO BE CHANGED SO SKMODELS!
    def skModels(self,model, X, y):
        # models we want to try
        names = ["knn", "linear svm", 
        "rbv svm", "gaussian process","descision tree", "random forest", 
        "nn", "ada boost","nb", "qda", 'logit']
        classifiers = [ KNeighborsClassifier(3),
                        SVC(kernel="linear", C=0.025),
                        SVC(gamma=2, C=1),
                        GaussianProcessClassifier(1.0 * RBF(1.0)),
                        DecisionTreeClassifier(max_depth=5),
                        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                        MLPClassifier(alpha=1, max_iter=1000),
                        AdaBoostClassifier(),
                        GaussianNB(),
                        QuadraticDiscriminantAnalysis()]
        model_types = [type(classifier) for classifier in classifiers]
        if model in names:
            dict_of_classifiers = dict(zip(names, classifiers))
            return  dict_of_classifiers, model_types   
        else:
            raise InputError('Unknown model type.')

    def fit_skModels(self, model_type, X,y):
        if model_type in dict_:
            fitted_model = dict_[model_type].fit(X,y)
            return fitted_model
        else:
            raise InputError('Unknown model type')
    
    def skModels_predict(self, model, X):
        import sklearn
        types = [sklearn.neighbors._classification.KNeighborsClassifier,
                sklearn.svm._classes.SVC,
                sklearn.svm._classes.SVC,
                sklearn.gaussian_process._gpc.GaussianProcessClassifier,
                sklearn.tree._classes.DecisionTreeClassifier,
                sklearn.ensemble._forest.RandomForestClassifier,
                sklearn.neural_network._multilayer_perceptron.MLPClassifier,
                sklearn.ensemble._weight_boosting.AdaBoostClassifier,
                sklearn.naive_bayes.GaussianNB,
                sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis]
        if type(model) in types:
            y_hat = model.predict(X)
        else:
            raise InputError('Unknown model type.')
        return y_hat

            
##############################################################################################################################
    def fit_smLogit(self, X, y):
      return sm.Logit(y, X).fit(disp=False)

    def fit(self, X, y, model_type):
      if model_type == 'logit':
          model = self.fit_smLogit(X, y)
      else:
        raise InputError('Unknown model type.')
      return model

    def predict(self, model, X):
      X = np.array(X)

      if type(model) == statsmodels.discrete.discrete_model.BinaryResultsWrapper:
          y_hat = model.predict(X)
      else:
          raise InputError('Unknown model type.')
      return y_hat

    def pipeline(self, df, date, model_type, train_start):
      X_train, y_train, X_test, y_test = self.train_test_split(df, date, train_start)
      model = self.fit(X_train, y_train, model_type)
      return self.predict(model, X_test)


    def auc(self, y_true, y_hat):
      import sklearn.metrics
      return sklearn.metrics.roc_auc_score(y_true, y_hat)


    def evaluate(self, df, model_type, train_start, predict_start='2014-01-01', predict_end=-1):
      dates = pd.DataFrame(df.index)
      dates = dates.set_index(df.index)['Time']
      predict_start = pd.to_datetime(predict_start)
      predict_end = pd.to_datetime(dates.iloc[predict_end]) if type(predict_end) == int else pd.to_datetime(predict_end)
      dates = dates.loc[predict_start:predict_end]
      y_true = []
      y_hat_train = {}
      y_hat_test = []
      auc_train_dict = {}
      auc_test = []

      for date in dates.index:
          # train test split
          #train_test_split(self, df, date, train_start='2013-11-01', test_delta='all', target='activity')
          X_train, y_train, X_test, y_test = self.train_test_split(df, date, train_start)

          # fit model
          model = self.fit(X_train, y_train, model_type)

          # predict
          y_hat_train.update({date: self.predict(model, X_train)})
          y_hat_test += list(self.predict(model, X_test))

          # evaluate train data
          auc_train_dict.update({date: self.auc(y_train, list(y_hat_train.values())[-1])})
        
          y_true += list(y_test)
    
      auc_test = self.auc(y_true, y_hat_test)
      auc_train = np.mean(list(auc_train_dict.values()))

      return auc_train, auc_test, auc_train_dict
