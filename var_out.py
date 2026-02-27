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
logger = setup_logging('var_out')
from scipy import stats

def variable_transformation_outliers(X_train_num,X_test_num):
    try:
        logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
        logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")

        for i in X_train_num.columns:
            X_train_num[i + "_yeo"],lam = stats.yeojohnson(X_train_num[i])
            X_train_num = X_train_num.drop([i],axis = 1)
            iqr = X_train_num[i+"_yeo"].quantile(0.75) - X_train_num[i+"_yeo"].quantile(0.25)
            upper_limit = X_train_num[i+"_yeo"].quantile(0.75)+(1.5*iqr)
            lower_limit = X_train_num[i+"_yeo"].quantile(0.25)-(1.5*iqr)
            X_train_num[i+"yeo_trim"] = np.where(X_train_num[i+"_yeo"]>upper_limit,upper_limit,
                     np.where(X_train_num[i+"_yeo"]<lower_limit,lower_limit,X_train_num[i+"_yeo"]))
            X_train_num = X_train_num.drop([i+"_yeo"],axis = 1)
            X_test_num[i + "yeo_trim"] = np.where(X_test_num[i] > upper_limit, upper_limit,
                                                   np.where(X_test_num[i]< lower_limit, lower_limit,
                                                            X_test_num[i]))
            X_test_num = X_test_num.drop([i], axis=1)

        logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
        logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")

        # for i in X_train_num.columns:
        #     sns.boxplot(x = X_train_num[i])
        #     plt.show()

        return X_train_num,X_test_num

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')