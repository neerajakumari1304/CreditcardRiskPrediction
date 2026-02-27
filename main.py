"""
in this file we are going to load the data and other ml pipeline techniques which are needed
"""
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
logger = setup_logging('main')
from sklearn.model_selection import train_test_split
from random_sample import random_sample_imputation_technique
from var_out import variable_transformation_outliers
from feature_selection import complete_feature_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from imbalance_data import balance_data

class CREDIT_CARD_DATA:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            #debug | info | critical | warning | error
            logger.info(f'Data loaded succesfully')
            logger.info(f'total rows in the data : {self.df.shape[0]}')
            logger.info(f'total columns in the data : {self.df.shape[1]}')
            logger.info(f'Before : {self.df.isnull().sum()}')
            self.df = self.df.drop([150000,150001],axis=0)
            self.df = self.df.drop(['MonthlyIncome.1'],axis=1)
            logger.info(f'========================================')
            logger.info(f'After : {self.df.isnull().sum()}')
            logger.info(f'========================================')
            for i in self.df.columns:
                if self.df[i].isnull().sum()>0:
                    logger.info(f'{i}->{self.df[i].dtype}')
                    if self.df[i].dtype == 'object':
                        self.df[i] = pd.to_numeric(self.df[i])
                        logger.info(f'{i}->{self.df[i].dtype}')
                    else:
                        pass
            self.X = self.df.iloc[:,:-1] #independent
            self.y = self.df.iloc[:,-1]  #dependent

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)




            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'training data size:{self.X_train.shape}')
            logger.info(f'testing data size:{self.X_test.shape}')

        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

    def missing_value(self):
        try:
            logger.info(f'total rows in training data: {self.X_train.shape}')
            logger.info(f'total rows in testing data: {self.X_test.shape}')
            logger.info(f"Before : {self.X_train.columns}")
            logger.info(f"Before : {self.X_test.columns}")
            logger.info(f"Before : {self.X_train.isnull().sum()}")
            logger.info(f"Before : {self.X_test.isnull().sum()}")
            self.X_train,self.X_test = random_sample_imputation_technique(self.X_train,self.X_test)
            logger.info(f"After : {self.X_train.columns}")
            logger.info(f"After : {self.X_test.columns}")
            logger.info(f"After : {self.X_train.isnull().sum()}")
            logger.info(f"After : {self.X_test.isnull().sum()}")
            logger.info(f'total rows in training data: {self.X_train.shape}')
            logger.info(f'total rows in testing data: {self.X_test.shape}')
        except Exception as e:
            error_type,error_msg,error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

    def vt_hol(self):
        try:
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')
            logger.info(f'-------------------------------------------')
            self.X_train_num = self.X_train.select_dtypes(exclude = 'object')
            self.X_train_cat = self.X_train.select_dtypes(include = 'object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f'{self.X_train_num.columns}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_test_cat.shape}')
            self.X_train_num,self.X_test_num = variable_transformation_outliers(self.X_train_num,self.X_test_num)
            logger.info(f"{self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"{self.X_test_num.columns} -> {self.X_test_num.shape}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

    def fs(self):
        try:
            logger.info(f"Before : {self.X_train_num.columns}->{self.X_train_num.shape}")
            logger.info(f"Before : {self.X_test_num.columns}->{self.X_test_num.shape}")
            self.X_train_num,self.X_test_num = complete_feature_selection(self.X_train_num,self.X_test_num,self.y_train)
            logger.info(f"After : {self.X_train_num.columns}->{self.X_train_num.shape}")
            logger.info(f"After : {self.X_test_num.columns}->{self.X_test_num.shape}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

    def cat_to_num(self):
        try:
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            for i in self.X_train_cat.columns:
                logger.info(f'{i}->:{self.X_train_cat[i].unique()}')

            logger.info(f"before converting:{self.X_train_cat}")
            logger.info(f"before converting:{self.X_test_cat}")


            one_hot = OneHotEncoder(drop='first')
            one_hot.fit(self.X_train_cat[["Gender","Region"]])
            result = one_hot.transform(self.X_train_cat[["Gender","Region"]]).toarray()
            f = pd.DataFrame(data=result, columns=one_hot.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f], axis=1)
            self.X_train_cat = self.X_train_cat.drop(["Gender","Region"],axis=1)

            result1 = one_hot.transform(self.X_test_cat[["Gender", "Region"]]).toarray()
            f1 = pd.DataFrame(data=result1, columns=one_hot.get_feature_names_out())
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)
            self.X_test_cat = pd.concat([self.X_test_cat, f1], axis=1)
            self.X_test_cat = self.X_test_cat.drop(["Gender", "Region"], axis=1)

            ord_end = OrdinalEncoder()
            ord_end.fit(self.X_train_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            result2 = ord_end.transform(self.X_train_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            t = pd.DataFrame(data=result2, columns=ord_end.get_feature_names_out() + '_res')
            self.X_train_cat.reset_index(drop=True, inplace=True)
            t.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, t], axis=1)
            self.X_train_cat = self.X_train_cat.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis=1)

            result3 = ord_end.transform(self.X_test_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            t1 = pd.DataFrame(data=result3, columns=ord_end.get_feature_names_out() + '_res')
            self.X_test_cat.reset_index(drop=True, inplace=True)
            t1.reset_index(drop=True, inplace=True)
            self.X_test_cat = pd.concat([self.X_test_cat, t1], axis=1)
            self.X_test_cat = self.X_test_cat.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis=1)

            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')

            logger.info(f"After Converting : {self.X_train_cat}")
            logger.info(f"After Converting : {self.X_test_cat}")

            logger.info(f"{self.X_train_cat.shape}")
            logger.info(f"{self.X_test_cat.shape}")

            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")

            self.X_train_num.reset_index(drop=True, inplace=True)
            self.X_train_cat.reset_index(drop=True, inplace=True)

            self.X_test_num.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)

            self.training_data = pd.concat([self.X_train_num,self.X_train_cat],axis=1)
            self.testing_data = pd.concat([self.X_test_num,self.X_test_cat],axis = 1)

            logger.info(f"{self.training_data.shape}")
            logger.info(f"{self.testing_data.shape}")

            logger.info(f"{self.training_data.isnull().sum()}")
            logger.info(f"{self.testing_data.isnull().sum()}")

            logger.info(f"================================================")

            logger.info(f"{self.training_data.sample(10)}")
            logger.info(f"{self.testing_data.sample(10)}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

    def data_b(self):
        try:
            self.y_train = self.y_train.map({'Good':1,'Bad':0}).astype(int)
            self.y_test =  self.y_test.map({'Good':1,'Bad':0}).astype(int)
            balance_data(self.training_data,self.y_train,self.testing_data,self.y_test)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')



if __name__ == "__main__":
    try:
        obj = CREDIT_CARD_DATA('D:\\ML_Pipeline_PROJECT\\ML\\creditcard.csv')
        obj.missing_value()
        obj.vt_hol()
        obj.fs()
        obj.cat_to_num()
        obj.data_b()
    except Exception as e:
        error_type,error_msg,error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

