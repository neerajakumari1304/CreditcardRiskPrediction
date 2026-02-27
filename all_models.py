import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('all_models')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
import pickle

def knn(X_train,y_train,X_test,y_test):
    try:
      global knn_reg
      knn_reg = KNeighborsClassifier(n_neighbors=5)
      knn_reg.fit(X_train,y_train)
      logger.info(f'KNN Test Accuracy : {accuracy_score(y_test,knn_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

def nb(X_train,y_train,X_test,y_test):
    try:
      global naive_reg
      naive_reg = GaussianNB()
      naive_reg.fit(X_train,y_train)
      logger.info(f'Naive Bayes Test Accuracy : {accuracy_score(y_test,naive_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

def lr(X_train,y_train,X_test,y_test):
    try:
      global lr_reg
      lr_reg = LogisticRegression()
      lr_reg.fit(X_train,y_train)
      logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test,lr_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

def dt(X_train,y_train,X_test,y_test):
    try:
      global dt_reg
      dt_reg = DecisionTreeClassifier(criterion='entropy')
      dt_reg.fit(X_train,y_train)
      logger.info(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(y_test,dt_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

def rf(X_train,y_train,X_test,y_test):
    try:
      global rf_reg
      rf_reg = RandomForestClassifier(n_estimators=5,criterion='entropy')
      rf_reg.fit(X_train,y_train)
      logger.info(f'RandomForestClassifier Test Accuracy : {accuracy_score(y_test,rf_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

def ada(X_train,y_train,X_test,y_test):
    try:
      global ada_reg
      t = LogisticRegression()
      ada_reg = AdaBoostClassifier(estimator=t,n_estimators=5)
      ada_reg.fit(X_train,y_train)
      logger.info(f'AdaBoostClassifier Test Accuracy : {accuracy_score(y_test,ada_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')


def gb(X_train,y_train,X_test,y_test):
    try:
      global gb_reg
      gb_reg = GradientBoostingClassifier(n_estimators=5)
      gb_reg.fit(X_train,y_train)
      logger.info(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(y_test,gb_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

def xgb_(X_train,y_train,X_test,y_test):
    try:
      global xg_reg
      xg_reg = XGBClassifier()
      xg_reg.fit(X_train,y_train)
      logger.info(f'XGBClassifier Test Accuracy : {accuracy_score(y_test,xg_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

def svm_c(X_train,y_train,X_test,y_test):
    try:
      global svm_reg
      svm_reg = SVC(kernel='rbf')
      svm_reg.fit(X_train,y_train)
      logger.info(f'SVM Test Accuracy : {accuracy_score(y_test,svm_reg.predict(X_test))}')

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')


def common(X_train,y_train,X_test,y_test):
    try:
      logger.info('=========KNN===========')
      knn(X_train,y_train,X_test,y_test)
      logger.info('=========NB===========')
      nb(X_train,y_train,X_test,y_test)
      logger.info('=========LR===========')
      lr(X_train,y_train,X_test,y_test)
      logger.info('=========DT===========')
      dt(X_train,y_train,X_test,y_test)
      logger.info('=========RF===========')
      rf(X_train,y_train,X_test,y_test)
      logger.info('=========Adabost===========')
      ada(X_train,y_train,X_test,y_test)
      logger.info('=========GB===========')
      gb(X_train,y_train,X_test,y_test)
      logger.info('=========XGB===========')
      xgb_(X_train,y_train,X_test,y_test)
      logger.info('=========svm===========')
      svm_c(X_train,y_train,X_test,y_test)
      knn_predictions = knn_reg.predict_proba(X_test)[:, 1]
      naive_predictions = naive_reg.predict_proba(X_test)[:, 1]
      lr_predictions = lr_reg.predict_proba(X_test)[:, 1]
      with open('creditcard.pkl','wb') as f:
          pickle.dump(lr_reg,f)
      dt_predictions = dt_reg.predict_proba(X_test)[:, 1]
      rf_predictions = rf_reg.predict_proba(X_test)[:, 1]
      ada_predictions = ada_reg.predict_proba(X_test)[:, 1]
      gb_predictions = gb_reg.predict_proba(X_test)[:, 1]
      xgb_predictions = xg_reg.predict_proba(X_test)[:, 1]
      svm_predictions = svm_reg.predict(X_test)
      knn_fpr, knn_tpr, knn_thre = roc_curve(y_test, knn_predictions)
      nb_fpr, nb_tpr, nb_thre = roc_curve(y_test, naive_predictions)
      lr_fpr, lr_tpr, lr_thre = roc_curve(y_test, lr_predictions)
      dt_fpr, dt_tpr, dt_thre = roc_curve(y_test, dt_predictions)
      rf_fpr, rf_tpr, rf_thre = roc_curve(y_test, rf_predictions)
      ada_fpr, ada_tpr, ada_thre = roc_curve(y_test, ada_predictions)
      gb_fpr, gb_tpr, gb_thre = roc_curve(y_test, gb_predictions)
      xgb_fpr, xgb_tpr, xgb_thre = roc_curve(y_test, xgb_predictions)
      svm_fpr, svm_tpr, svm_thre = roc_curve(y_test, svm_predictions)

      plt.figure(figsize=(5, 3))
      plt.plot([0, 1], [0, 1], "k--")

      plt.plot(knn_fpr, knn_tpr, label="KNN")
      plt.plot(nb_fpr, nb_tpr, label="NB")
      plt.plot(lr_fpr, lr_tpr, label="LR")
      plt.plot(dt_fpr, dt_tpr, label="DT")
      plt.plot(rf_fpr, rf_tpr, label="RF")
      plt.plot(ada_fpr, ada_tpr, label="ADA")
      plt.plot(gb_fpr, gb_tpr, label="GB")
      plt.plot(xgb_fpr, xgb_tpr, label="XGB")
      plt.plot(svm_fpr, svm_tpr, label="SVM")

      plt.xlabel("FPR")
      plt.ylabel("TPR")
      plt.title("ROC Curve - ALL Models")
      plt.legend(loc=0)
      plt.show()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')