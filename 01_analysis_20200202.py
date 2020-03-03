"""
TITLE: "Bayesian Networks exact linear regression"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'

"""

###############################################################################
## 1. IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import numpy as np
import pandas as pd
from copy import deepcopy
import pymc3 as pm
import joblib
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelEncoder  
#from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import time
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


###############################################################################
## 2. SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.4.0-143-generic').
RELEASE = platform.release()

if RELEASE == '5.3.0-40-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/bayesian_networks/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/bayesian_networks/outputs')

else:
   BASE_DIR_INPUT = ('/home/ubuntu/raw_data')
   BASE_DIR_OUTPUT = ('/home/ubuntu/outputs')

   
###############################################################################
## 3. PARAMETERS TO BE SET!!!
input_file_name_1 = ('PCA_Prep.csv')
output_file_name_6 = None 
output_file_name_2 = ('012_analysis_bayesian_model.joblib') 
output_file_name_3 = ('012_analysis_trace_model.joblib') 
output_file_name_4 = ('012_analysis_X_test_oob.csv') 
output_file_name_5 = ('012_analysis_y_test_oob.csv')
output_file_name_7 = ('012_analysis_saved_trace')


# setting PyMC3 parameters (IDEAL)
# ideal: 440000 draws
draws = 1000
chains = 7 # IDEAL: many chain as many cores
tune = 3000
#tune = (draws*90)/100 # ideal: 90 % burn-in (also called "tune")
cores = int(round((cpu_count() - 1), 0))

# start clocking time
start_time = time.time()


###############################################################################
## 4. LOADING DATA-SET 
if output_file_name_6 is not None:
    # loading the .csv file 
    X_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                     output_file_name_6]), header = 0) 
#elif input_file_name_1 is not None: 
else: 
    # loading the .csv file 
    X_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                 input_file_name_1]), header = 0)

    
###############################################################################
## 6. PRE-PROCESSING 
# deepcopy
X = X_tmp.copy()    
    
# TEST: double-check columns' names
#X_columns_1 = pd.DataFrame(X.columns)

# drop rows with NaN
X.dropna(axis = 0, inplace = True)

# drop duplicates
X.drop_duplicates(inplace = True)

# split data-set in train set and out-of-bag set after randomly sampling it
X_test_oob = X.sample(frac = 0.2)
X.drop(index = X_test_oob.index, inplace = True)

# save not standardized train hold-out-sets as .csv files
X_test_oob.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_4]), index= False)

# free-up RAM memory
X_tmp = None

# deep copy
X_train = X.copy() 

## standardize necessary step, otherwise PyMC3 throws errors)
ss_train = StandardScaler()
X_train.loc[:, :] = ss_train.fit_transform(X_train.loc[:, :])  


###############################################################################
## 8. BAYESIAN + MCMC LINEAR REGRESSION  

# set shared theano variables
x_01_data = X_train.loc[:, 'var_01'].to_numpy()
x_02_data = X_train.loc[:, 'var_02'].to_numpy()
x_03_data = X_train.loc[:, 'var_03'].to_numpy()
x_04_data = X_train.loc[:, 'var_04'].to_numpy()
x_05_data = X_train.loc[:, 'var_05'].to_numpy()
x_06_data = X_train.loc[:, 'var_06'].to_numpy()
x_07_data = X_train.loc[:, 'var_07'].to_numpy()
x_08_data = X_train.loc[:, 'var_08'].to_numpy()


###############################################################################
## VARYING_INTERCEPT_AND_SLOPE (NON-CENTERED)

def model_factory(x_01_data, x_02_data, x_03_data, x_04_data, x_05_data, 
                  x_06_data, x_07_data, x_08_data):
    
    with pm.Model() as varying_intercept_slope_noncentered:
        
       
        # Hyperpriors of intercepts for connections
        mu_alpha_node_01_edges = pm.Normal('mu_alpha_node_01_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_01_edges = pm.HalfCauchy('sigma_alpha_node_01_edges', 5)
        mu_alpha_node_02_edges = pm.Normal('mu_alpha_node_02_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_02_edges = pm.HalfCauchy('sigma_alpha_node_02_edges', 5)
        mu_alpha_node_03_edges = pm.Normal('mu_alpha_node_03_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_03_edges = pm.HalfCauchy('sigma_alpha_node_03_edges', 5)
        mu_alpha_node_04_edges = pm.Normal('mu_alpha_node_04_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_04_edges = pm.HalfCauchy('sigma_alpha_node_04_edges', 5)
        mu_alpha_node_05_edges = pm.Normal('mu_alpha_node_05_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_05_edges = pm.HalfCauchy('sigma_alpha_node_05_edges', 5)
        mu_alpha_node_06_edges = pm.Normal('mu_alpha_node_06_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_06_edges = pm.HalfCauchy('sigma_alpha_node_06_edges', 5)
        mu_alpha_node_07_edges = pm.Normal('mu_alpha_node_07_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_07_edges = pm.HalfCauchy('sigma_alpha_node_07_edges', 5)
        mu_alpha_node_08_edges = pm.Normal('mu_alpha_node_08_edges', mu = 0.05, sigma = 2)
        sigma_alpha_node_08_edges = pm.HalfCauchy('sigma_alpha_node_08_edges', 5)
        
        # Hyperpriors of slopes for connections
        # node_01 connections
        mu_beta_node_01_node_02 = pm.Normal('mu_beta_node_01_node_02', mu = 0.05, sigma = 2)
        sigma_beta_node_01_node_02 = pm.HalfCauchy('sigma_beta_node_01_node_02', 5)
        mu_beta_node_01_node_03 = pm.Normal('mu_beta_node_01_node_03', mu = 0.05, sigma = 2)
        sigma_beta_node_01_node_03 = pm.HalfCauchy('sigma_beta_node_01_node_03', 5)
        mu_beta_node_01_node_04 = pm.Normal('mu_beta_node_01_node_04', mu = 0.05, sigma = 2)
        sigma_beta_node_01_node_04 = pm.HalfCauchy('sigma_beta_node_01_node_04', 5)
        mu_beta_node_01_node_05 = pm.Normal('mu_beta_node_01_node_05', mu = 0.05, sigma = 2)
        sigma_beta_node_01_node_05 = pm.HalfCauchy('sigma_beta_node_01_node_05', 5)
        mu_beta_node_01_node_06 = pm.Normal('mu_beta_node_01_node_06', mu = 0.05, sigma = 2)
        sigma_beta_node_01_node_06 = pm.HalfCauchy('sigma_beta_node_01_node_06', 5)
        mu_beta_node_01_node_07 = pm.Normal('mu_beta_node_01_node_07', mu = 0.05, sigma = 2)
        sigma_beta_node_01_node_07 = pm.HalfCauchy('sigma_beta_node_01_node_07', 5)
        mu_beta_node_01_node_08 = pm.Normal('mu_beta_node_01_node_08', mu = 0.05, sigma = 2)
        sigma_beta_node_01_node_08 = pm.HalfCauchy('sigma_beta_node_01_node_08', 5)
        # node_02 connections
        mu_beta_node_02_node_01 = pm.Normal('mu_beta_node_02_node_01', mu = 0.05, sigma = 2)
        sigma_beta_node_02_node_01 = pm.HalfCauchy('sigma_beta_node_02_node_01', 5)
        mu_beta_node_02_node_03 = pm.Normal('mu_beta_node_02_node_03', mu = 0.05, sigma = 2)
        sigma_beta_node_02_node_03 = pm.HalfCauchy('sigma_beta_node_02_node_03', 5)
        mu_beta_node_02_node_04 = pm.Normal('mu_beta_node_02_node_04', mu = 0.05, sigma = 2)
        sigma_beta_node_02_node_04 = pm.HalfCauchy('sigma_beta_node_02_node_04', 5)
        mu_beta_node_02_node_05 = pm.Normal('mu_beta_node_02_node_05', mu = 0.05, sigma = 2)
        sigma_beta_node_02_node_05 = pm.HalfCauchy('sigma_beta_node_02_node_05', 5)
        mu_beta_node_02_node_06 = pm.Normal('mu_beta_node_02_node_06', mu = 0.05, sigma = 2)
        sigma_beta_node_02_node_06 = pm.HalfCauchy('sigma_beta_node_02_node_06', 5)
        mu_beta_node_02_node_07 = pm.Normal('mu_beta_node_02_node_07', mu = 0.05, sigma = 2)
        sigma_beta_node_02_node_07 = pm.HalfCauchy('sigma_beta_node_02_node_07', 5)
        mu_beta_node_02_node_08 = pm.Normal('mu_beta_node_02_node_08', mu = 0.05, sigma = 2)
        sigma_beta_node_02_node_08 = pm.HalfCauchy('sigma_beta_node_02_node_08', 5)
        # node_03 connections
        mu_beta_node_03_node_01 = pm.Normal('mu_beta_node_03_node_01', mu = 0.05, sigma = 2)
        sigma_beta_node_03_node_01 = pm.HalfCauchy('sigma_beta_node_03_node_01', 5)
        mu_beta_node_03_node_02 = pm.Normal('mu_beta_node_03_node_02', mu = 0.05, sigma = 2)
        sigma_beta_node_03_node_02 = pm.HalfCauchy('sigma_beta_node_03_node_02', 5)
        mu_beta_node_03_node_04 = pm.Normal('mu_beta_node_03_node_04', mu = 0.05, sigma = 2)
        sigma_beta_node_03_node_04 = pm.HalfCauchy('sigma_beta_node_03_node_04', 5)
        mu_beta_node_03_node_05 = pm.Normal('mu_beta_node_03_node_05', mu = 0.05, sigma = 2)
        sigma_beta_node_03_node_05 = pm.HalfCauchy('sigma_beta_node_03_node_05', 5)
        mu_beta_node_03_node_06 = pm.Normal('mu_beta_node_03_node_06', mu = 0.05, sigma = 2)
        sigma_beta_node_03_node_06 = pm.HalfCauchy('sigma_beta_node_03_node_06', 5)
        mu_beta_node_03_node_07 = pm.Normal('mu_beta_node_03_node_07', mu = 0.05, sigma = 2)
        sigma_beta_node_03_node_07 = pm.HalfCauchy('sigma_beta_node_03_node_07', 5)
        mu_beta_node_03_node_08 = pm.Normal('mu_beta_node_03_node_08', mu = 0.05, sigma = 2)
        sigma_beta_node_03_node_08 = pm.HalfCauchy('sigma_beta_node_03_node_08', 5)
        # node_04 connections
        mu_beta_node_04_node_01 = pm.Normal('mu_beta_node_04_node_01', mu = 0.05, sigma = 2)
        sigma_beta_node_04_node_01 = pm.HalfCauchy('sigma_beta_node_04_node_01', 5)
        mu_beta_node_04_node_02 = pm.Normal('mu_beta_node_04_node_02', mu = 0.05, sigma = 2)
        sigma_beta_node_04_node_02 = pm.HalfCauchy('sigma_beta_node_04_node_02', 5)
        mu_beta_node_04_node_03 = pm.Normal('mu_beta_node_04_node_03', mu = 0.05, sigma = 2)
        sigma_beta_node_04_node_03 = pm.HalfCauchy('sigma_beta_node_04_node_03', 5)
        mu_beta_node_04_node_05 = pm.Normal('mu_beta_node_04_node_05', mu = 0.05, sigma = 2)
        sigma_beta_node_04_node_05 = pm.HalfCauchy('sigma_beta_node_04_node_05', 5)
        mu_beta_node_04_node_06 = pm.Normal('mu_beta_node_04_node_06', mu = 0.05, sigma = 2)
        sigma_beta_node_04_node_06 = pm.HalfCauchy('sigma_beta_node_04_node_06', 5)
        mu_beta_node_04_node_07 = pm.Normal('mu_beta_node_04_node_07', mu = 0.05, sigma = 2)
        sigma_beta_node_04_node_07 = pm.HalfCauchy('sigma_beta_node_04_node_07', 5)
        mu_beta_node_04_node_08 = pm.Normal('mu_beta_node_04_node_08', mu = 0.05, sigma = 2)
        sigma_beta_node_04_node_08 = pm.HalfCauchy('sigma_beta_node_04_node_08', 5)
        # node_05 connections
        mu_beta_node_05_node_01 = pm.Normal('mu_beta_node_05_node_01', mu = 0.05, sigma = 2)
        sigma_beta_node_05_node_01 = pm.HalfCauchy('sigma_beta_node_05_node_01', 5)
        mu_beta_node_05_node_02 = pm.Normal('mu_beta_node_05_node_02', mu = 0.05, sigma = 2)
        sigma_beta_node_05_node_02 = pm.HalfCauchy('sigma_beta_node_05_node_02', 5)
        mu_beta_node_05_node_03 = pm.Normal('mu_beta_node_05_node_03', mu = 0.05, sigma = 2)
        sigma_beta_node_05_node_03 = pm.HalfCauchy('sigma_beta_node_05_node_03', 5)
        mu_beta_node_05_node_04 = pm.Normal('mu_beta_node_05_node_04', mu = 0.05, sigma = 2)
        sigma_beta_node_05_node_04 = pm.HalfCauchy('sigma_beta_node_05_node_04', 5)
        mu_beta_node_05_node_06 = pm.Normal('mu_beta_node_05_node_06', mu = 0.05, sigma = 2)
        sigma_beta_node_05_node_06 = pm.HalfCauchy('sigma_beta_node_05_node_06', 5)
        mu_beta_node_05_node_07 = pm.Normal('mu_beta_node_05_node_07', mu = 0.05, sigma = 2)
        sigma_beta_node_05_node_07 = pm.HalfCauchy('sigma_beta_node_05_node_07', 5)
        mu_beta_node_05_node_08 = pm.Normal('mu_beta_node_05_node_08', mu = 0.05, sigma = 2)
        sigma_beta_node_05_node_08 = pm.HalfCauchy('sigma_beta_node_05_node_08', 5)
        # node_06 connections
        mu_beta_node_06_node_01 = pm.Normal('mu_beta_node_06_node_01', mu = 0.05, sigma = 2)
        sigma_beta_node_06_node_01 = pm.HalfCauchy('sigma_beta_node_06_node_01', 5)
        mu_beta_node_06_node_02 = pm.Normal('mu_beta_node_06_node_02', mu = 0.05, sigma = 2)
        sigma_beta_node_06_node_02 = pm.HalfCauchy('sigma_beta_node_06_node_02', 5)
        mu_beta_node_06_node_03 = pm.Normal('mu_beta_node_06_node_03', mu = 0.05, sigma = 2)
        sigma_beta_node_06_node_03 = pm.HalfCauchy('sigma_beta_node_06_node_03', 5)
        mu_beta_node_06_node_04 = pm.Normal('mu_beta_node_06_node_04', mu = 0.05, sigma = 2)
        sigma_beta_node_06_node_04 = pm.HalfCauchy('sigma_beta_node_06_node_04', 5)
        mu_beta_node_06_node_05 = pm.Normal('mu_beta_node_06_node_05', mu = 0.05, sigma = 2)
        sigma_beta_node_06_node_05 = pm.HalfCauchy('sigma_beta_node_06_node_05', 5)
        mu_beta_node_06_node_07 = pm.Normal('mu_beta_node_06_node_07', mu = 0.05, sigma = 2)
        sigma_beta_node_06_node_07 = pm.HalfCauchy('sigma_beta_node_06_node_07', 5)
        mu_beta_node_06_node_08 = pm.Normal('mu_beta_node_06_node_08', mu = 0.05, sigma = 2)
        sigma_beta_node_06_node_08 = pm.HalfCauchy('sigma_beta_node_06_node_08', 5)
        # node_07 connections
        mu_beta_node_07_node_01 = pm.Normal('mu_beta_node_07_node_01', mu = 0.05, sigma = 2)
        sigma_beta_node_07_node_01 = pm.HalfCauchy('sigma_beta_node_07_node_01', 5)
        mu_beta_node_07_node_02 = pm.Normal('mu_beta_node_07_node_02', mu = 0.05, sigma = 2)
        sigma_beta_node_07_node_02 = pm.HalfCauchy('sigma_beta_node_07_node_02', 5)
        mu_beta_node_07_node_03 = pm.Normal('mu_beta_node_07_node_03', mu = 0.05, sigma = 2)
        sigma_beta_node_07_node_03 = pm.HalfCauchy('sigma_beta_node_07_node_03', 5)
        mu_beta_node_07_node_04 = pm.Normal('mu_beta_node_07_node_04', mu = 0.05, sigma = 2)
        sigma_beta_node_07_node_04 = pm.HalfCauchy('sigma_beta_node_07_node_04', 5)
        mu_beta_node_07_node_05 = pm.Normal('mu_beta_node_07_node_05', mu = 0.05, sigma = 2)
        sigma_beta_node_07_node_05 = pm.HalfCauchy('sigma_beta_node_07_node_05', 5)
        mu_beta_node_07_node_06 = pm.Normal('mu_beta_node_07_node_06', mu = 0.05, sigma = 2)
        sigma_beta_node_07_node_06 = pm.HalfCauchy('sigma_beta_node_07_node_06', 5)
        mu_beta_node_07_node_08 = pm.Normal('mu_beta_node_07_node_08', mu = 0.05, sigma = 2)
        sigma_beta_node_07_node_08 = pm.HalfCauchy('sigma_beta_node_07_node_08', 5)
        # node_08 connections
        mu_beta_node_08_node_01 = pm.Normal('mu_beta_node_08_node_01', mu = 0.05, sigma = 2)
        sigma_beta_node_08_node_01 = pm.HalfCauchy('sigma_beta_node_08_node_01', 5)
        mu_beta_node_08_node_02 = pm.Normal('mu_beta_node_08_node_02', mu = 0.05, sigma = 2)
        sigma_beta_node_08_node_02 = pm.HalfCauchy('sigma_beta_node_08_node_02', 5)
        mu_beta_node_08_node_03 = pm.Normal('mu_beta_node_08_node_03', mu = 0.05, sigma = 2)
        sigma_beta_node_08_node_03 = pm.HalfCauchy('sigma_beta_node_08_node_03', 5)
        mu_beta_node_08_node_04 = pm.Normal('mu_beta_node_08_node_04', mu = 0.05, sigma = 2)
        sigma_beta_node_08_node_04 = pm.HalfCauchy('sigma_beta_node_08_node_04', 5)
        mu_beta_node_08_node_05 = pm.Normal('mu_beta_node_08_node_05', mu = 0.05, sigma = 2)
        sigma_beta_node_08_node_05 = pm.HalfCauchy('sigma_beta_node_08_node_05', 5)
        mu_beta_node_08_node_06 = pm.Normal('mu_beta_node_08_node_06', mu = 0.05, sigma = 2)
        sigma_beta_node_08_node_06 = pm.HalfCauchy('sigma_beta_node_08_node_06', 5)
        mu_beta_node_08_node_07 = pm.Normal('mu_beta_node_08_node_07', mu = 0.05, sigma = 2)
        sigma_beta_node_08_node_07 = pm.HalfCauchy('sigma_beta_node_08_node_07', 5)

        
        # node_01 connections
        alpha_node_01_edges = pm.Normal('alpha_node_01_edges', mu = mu_alpha_node_01_edges, sigma = sigma_alpha_node_01_edges)
        beta_node_01_node_02 = pm.Normal('beta_node_01_node_02', mu = mu_beta_node_01_node_02, sigma = sigma_beta_node_01_node_02)
        beta_node_01_node_03 = pm.Normal('beta_node_01_node_03', mu = mu_beta_node_01_node_03, sigma = sigma_beta_node_01_node_03)
        beta_node_01_node_04 = pm.Normal('beta_node_01_node_04', mu = mu_beta_node_01_node_04, sigma = sigma_beta_node_01_node_04)
        beta_node_01_node_05 = pm.Normal('beta_node_01_node_05', mu = mu_beta_node_01_node_05, sigma = sigma_beta_node_01_node_05)
        beta_node_01_node_06 = pm.Normal('beta_node_01_node_06', mu = mu_beta_node_01_node_06, sigma = sigma_beta_node_01_node_06)
        beta_node_01_node_07 = pm.Normal('beta_node_01_node_07', mu = mu_beta_node_01_node_07, sigma = sigma_beta_node_01_node_07)
        beta_node_01_node_08 = pm.Normal('beta_node_01_node_08', mu = mu_beta_node_01_node_08, sigma = sigma_beta_node_01_node_08)
        
        # node_02 connections
        alpha_node_02_edges = pm.Normal('alpha_node_02_edges', mu = mu_alpha_node_02_edges, sigma = sigma_alpha_node_02_edges)
        beta_node_02_node_01 = pm.Normal('beta_node_02_node_01', mu = mu_beta_node_02_node_01, sigma = sigma_beta_node_02_node_01)
        beta_node_02_node_03 = pm.Normal('beta_node_02_node_03', mu = mu_beta_node_02_node_03, sigma = sigma_beta_node_02_node_03)
        beta_node_02_node_04 = pm.Normal('beta_node_02_node_04', mu = mu_beta_node_02_node_04, sigma = sigma_beta_node_02_node_04)
        beta_node_02_node_05 = pm.Normal('beta_node_02_node_05', mu = mu_beta_node_02_node_05, sigma = sigma_beta_node_02_node_05)
        beta_node_02_node_06 = pm.Normal('beta_node_02_node_06', mu = mu_beta_node_02_node_06, sigma = sigma_beta_node_02_node_06)
        beta_node_02_node_07 = pm.Normal('beta_node_02_node_07', mu = mu_beta_node_02_node_07, sigma = sigma_beta_node_02_node_07)
        beta_node_02_node_08 = pm.Normal('beta_node_02_node_08', mu = mu_beta_node_02_node_08, sigma = sigma_beta_node_02_node_08)
        
        # node_03 connections
        alpha_node_03_edges = pm.Normal('alpha_node_03_edges', mu = mu_alpha_node_03_edges, sigma = sigma_alpha_node_03_edges)
        beta_node_03_node_01 = pm.Normal('beta_node_03_node_01', mu = mu_beta_node_03_node_01, sigma = sigma_beta_node_03_node_01)
        beta_node_03_node_02 = pm.Normal('beta_node_03_node_02', mu = mu_beta_node_03_node_02, sigma = sigma_beta_node_03_node_02)
        beta_node_03_node_04 = pm.Normal('beta_node_03_node_04', mu = mu_beta_node_03_node_04, sigma = sigma_beta_node_03_node_04)
        beta_node_03_node_05 = pm.Normal('beta_node_03_node_05', mu = mu_beta_node_03_node_05, sigma = sigma_beta_node_03_node_05)
        beta_node_03_node_06 = pm.Normal('beta_node_03_node_06', mu = mu_beta_node_03_node_06, sigma = sigma_beta_node_03_node_06)
        beta_node_03_node_07 = pm.Normal('beta_node_03_node_07', mu = mu_beta_node_03_node_07, sigma = sigma_beta_node_03_node_07)
        beta_node_03_node_08 = pm.Normal('beta_node_03_node_08', mu = mu_beta_node_03_node_08, sigma = sigma_beta_node_03_node_08)
        
        # node_04 connections
        alpha_node_04_edges = pm.Normal('alpha_node_04_edges', mu = mu_alpha_node_04_edges, sigma = sigma_alpha_node_04_edges)
        beta_node_04_node_01 = pm.Normal('beta_node_04_node_01', mu = mu_beta_node_04_node_01, sigma = sigma_beta_node_04_node_01)
        beta_node_04_node_02 = pm.Normal('beta_node_04_node_02', mu = mu_beta_node_04_node_02, sigma = sigma_beta_node_04_node_02)
        beta_node_04_node_03 = pm.Normal('beta_node_04_node_03', mu = mu_beta_node_04_node_03, sigma = sigma_beta_node_04_node_03)
        beta_node_04_node_05 = pm.Normal('beta_node_04_node_05', mu = mu_beta_node_04_node_05, sigma = sigma_beta_node_04_node_05)
        beta_node_04_node_06 = pm.Normal('beta_node_04_node_06', mu = mu_beta_node_04_node_06, sigma = sigma_beta_node_04_node_06)
        beta_node_04_node_07 = pm.Normal('beta_node_04_node_07', mu = mu_beta_node_04_node_07, sigma = sigma_beta_node_04_node_07)
        beta_node_04_node_08 = pm.Normal('beta_node_04_node_08', mu = mu_beta_node_04_node_08, sigma = sigma_beta_node_04_node_08)
        
        # node_05 connections
        alpha_node_05_edges = pm.Normal('alpha_node_05_edges', mu = mu_alpha_node_05_edges, sigma = sigma_alpha_node_05_edges)
        beta_node_05_node_01 = pm.Normal('beta_node_05_node_01', mu = mu_beta_node_05_node_01, sigma = sigma_beta_node_05_node_01)
        beta_node_05_node_02 = pm.Normal('beta_node_05_node_02', mu = mu_beta_node_05_node_02, sigma = sigma_beta_node_05_node_02)
        beta_node_05_node_03 = pm.Normal('beta_node_05_node_03', mu = mu_beta_node_05_node_03, sigma = sigma_beta_node_05_node_03)
        beta_node_05_node_04 = pm.Normal('beta_node_05_node_04', mu = mu_beta_node_05_node_04, sigma = sigma_beta_node_05_node_04)
        beta_node_05_node_06 = pm.Normal('beta_node_05_node_06', mu = mu_beta_node_05_node_06, sigma = sigma_beta_node_05_node_06)
        beta_node_05_node_07 = pm.Normal('beta_node_05_node_07', mu = mu_beta_node_05_node_07, sigma = sigma_beta_node_05_node_07)
        beta_node_05_node_08 = pm.Normal('beta_node_05_node_08', mu = mu_beta_node_05_node_08, sigma = sigma_beta_node_05_node_08)
        
        # node_06 connections
        alpha_node_06_edges = pm.Normal('alpha_node_06_edges', mu = mu_alpha_node_06_edges, sigma = sigma_alpha_node_06_edges)
        beta_node_06_node_01 = pm.Normal('beta_node_06_node_01', mu = mu_beta_node_06_node_01, sigma = sigma_beta_node_06_node_01)
        beta_node_06_node_02 = pm.Normal('beta_node_06_node_02', mu = mu_beta_node_06_node_02, sigma = sigma_beta_node_06_node_02)
        beta_node_06_node_03 = pm.Normal('beta_node_06_node_03', mu = mu_beta_node_06_node_03, sigma = sigma_beta_node_06_node_03)
        beta_node_06_node_04 = pm.Normal('beta_node_06_node_04', mu = mu_beta_node_06_node_04, sigma = sigma_beta_node_06_node_04)
        beta_node_06_node_05 = pm.Normal('beta_node_06_node_05', mu = mu_beta_node_06_node_05, sigma = sigma_beta_node_06_node_05)
        beta_node_06_node_07 = pm.Normal('beta_node_06_node_07', mu = mu_beta_node_06_node_07, sigma = sigma_beta_node_06_node_07)
        beta_node_06_node_08 = pm.Normal('beta_node_06_node_08', mu = mu_beta_node_06_node_08, sigma = sigma_beta_node_06_node_08)
        
        # node_07 connections
        alpha_node_07_edges = pm.Normal('alpha_node_07_edges', mu = mu_alpha_node_07_edges, sigma = sigma_alpha_node_07_edges)
        beta_node_07_node_01 = pm.Normal('beta_node_07_node_01', mu = mu_beta_node_07_node_01, sigma = sigma_beta_node_07_node_01)
        beta_node_07_node_02 = pm.Normal('beta_node_07_node_02', mu = mu_beta_node_07_node_02, sigma = sigma_beta_node_07_node_02)
        beta_node_07_node_03 = pm.Normal('beta_node_07_node_03', mu = mu_beta_node_07_node_03, sigma = sigma_beta_node_07_node_03)
        beta_node_07_node_04 = pm.Normal('beta_node_07_node_04', mu = mu_beta_node_07_node_04, sigma = sigma_beta_node_07_node_04)
        beta_node_07_node_05 = pm.Normal('beta_node_07_node_05', mu = mu_beta_node_07_node_05, sigma = sigma_beta_node_07_node_05)
        beta_node_07_node_06 = pm.Normal('beta_node_07_node_06', mu = mu_beta_node_07_node_06, sigma = sigma_beta_node_07_node_06)
        beta_node_07_node_08 = pm.Normal('beta_node_07_node_08', mu = mu_beta_node_07_node_08, sigma = sigma_beta_node_07_node_08)
        
        # node_08 connections
        alpha_node_08_edges = pm.Normal('alpha_node_08_edges', mu = mu_alpha_node_08_edges, sigma = sigma_alpha_node_08_edges)
        beta_node_08_node_01 = pm.Normal('beta_node_08_node_01', mu = mu_beta_node_08_node_01, sigma = sigma_beta_node_08_node_01)
        beta_node_08_node_02 = pm.Normal('beta_node_08_node_02', mu = mu_beta_node_08_node_02, sigma = sigma_beta_node_08_node_02)
        beta_node_08_node_03 = pm.Normal('beta_node_08_node_03', mu = mu_beta_node_08_node_03, sigma = sigma_beta_node_08_node_03)
        beta_node_08_node_04 = pm.Normal('beta_node_08_node_04', mu = mu_beta_node_08_node_04, sigma = sigma_beta_node_08_node_04)
        beta_node_08_node_05 = pm.Normal('beta_node_08_node_05', mu = mu_beta_node_08_node_05, sigma = sigma_beta_node_08_node_05)
        beta_node_08_node_06 = pm.Normal('beta_node_08_node_06', mu = mu_beta_node_08_node_06, sigma = sigma_beta_node_08_node_06)
        beta_node_08_node_07 = pm.Normal('beta_node_08_node_07', mu = mu_beta_node_08_node_07, sigma = sigma_beta_node_08_node_07)
             
        # error term of the linear regression
        sigma_node_01_edges = pm.HalfCauchy('sigma_node_01_edges', 5)
        sigma_node_02_edges = pm.HalfCauchy('sigma_node_02_edges', 5)
        sigma_node_03_edges = pm.HalfCauchy('sigma_node_03_edges', 5)
        sigma_node_04_edges = pm.HalfCauchy('sigma_node_04_edges', 5)
        sigma_node_05_edges = pm.HalfCauchy('sigma_node_05_edges', 5)
        sigma_node_06_edges = pm.HalfCauchy('sigma_node_06_edges', 5)
        sigma_node_07_edges = pm.HalfCauchy('sigma_node_07_edges', 5) 
        sigma_node_08_edges = pm.HalfCauchy('sigma_node_08_edges', 5)
                             
        # expected value
        node_01_edges = pm.Normal('node_01_edges', mu = (alpha_node_01_edges + 
                                  beta_node_01_node_02*x_02_data + beta_node_01_node_03*x_03_data +
                           beta_node_01_node_04*x_04_data + beta_node_01_node_05*x_05_data + 
                           beta_node_01_node_06*x_06_data + beta_node_01_node_07*x_07_data +
                           beta_node_01_node_08*x_08_data), 
                           sigma = sigma_node_01_edges, observed = x_01_data)
        node_02_edges = pm.Normal('node_02_edges', mu = (alpha_node_02_edges + 
                                  beta_node_02_node_01*x_01_data + beta_node_02_node_03*x_03_data +
                           beta_node_02_node_04*x_04_data + beta_node_02_node_05*x_05_data + 
                           beta_node_02_node_06*x_06_data + beta_node_02_node_07*x_07_data +
                           beta_node_02_node_08*x_08_data),
                           sigma = sigma_node_02_edges, observed = x_02_data)
        node_03_edges = pm.Normal('node_03_edges', mu = (alpha_node_03_edges +
                           beta_node_03_node_01*x_01_data + beta_node_03_node_02*x_02_data +                             
                           beta_node_03_node_04*x_04_data + beta_node_03_node_05*x_05_data + 
                           beta_node_03_node_06*x_06_data + beta_node_03_node_07*x_07_data +
                           beta_node_03_node_08*x_08_data), 
                           sigma = sigma_node_03_edges, observed = x_03_data)
        node_04_edges = pm.Normal('node_04_edges', mu = (alpha_node_04_edges +
                           beta_node_04_node_01*x_01_data + beta_node_04_node_02*x_02_data +
                           beta_node_04_node_03*x_03_data + beta_node_04_node_05*x_05_data +
                           beta_node_04_node_06*x_06_data + beta_node_04_node_07*x_07_data +
                           beta_node_04_node_08*x_08_data), 
                           sigma = sigma_node_04_edges, observed = x_04_data)
        node_05_edges = pm.Normal('node_05_edges', mu = (alpha_node_05_edges + 
                           beta_node_05_node_01*x_01_data + beta_node_05_node_02*x_02_data +
                           beta_node_05_node_03*x_03_data + beta_node_05_node_04*x_04_data +
                           beta_node_05_node_06*x_06_data + beta_node_05_node_07*x_07_data +
                           beta_node_05_node_08*x_08_data), 
                           sigma = sigma_node_05_edges, observed = x_05_data)
        node_06_edges = pm.Normal('node_06_edges', mu = (alpha_node_06_edges + 
                           beta_node_06_node_01*x_01_data + beta_node_06_node_02*x_02_data +
                           beta_node_06_node_03*x_03_data + beta_node_06_node_04*x_04_data +
                           beta_node_06_node_05*x_05_data + beta_node_06_node_07*x_07_data +
                           beta_node_06_node_08*x_08_data), 
                           sigma = sigma_node_06_edges, observed = x_06_data)
        node_07_edges = pm.Normal('node_07_edges', mu = (alpha_node_07_edges +  
                           beta_node_07_node_01*x_01_data + beta_node_07_node_02*x_02_data +
                           beta_node_07_node_03*x_03_data + beta_node_07_node_04*x_04_data +
                           beta_node_07_node_05*x_05_data + beta_node_07_node_06*x_06_data +
                           beta_node_07_node_08*x_08_data), 
                           sigma = sigma_node_07_edges, observed = x_07_data)
    
        node_08_edges = pm.Normal('node_08_edges', mu = (alpha_node_08_edges +  
                           beta_node_08_node_01*x_01_data + beta_node_08_node_02*x_02_data +
                           beta_node_08_node_03*x_03_data + beta_node_08_node_04*x_04_data +
                           beta_node_08_node_05*x_05_data + beta_node_08_node_06*x_06_data +
                           beta_node_08_node_07*x_07_data), 
                           sigma = sigma_node_08_edges, observed = x_08_data)
       
        
    # dump trace model     
    joblib.dump(varying_intercept_slope_noncentered, os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2]))   
            
    return varying_intercept_slope_noncentered


with model_factory(x_01_data, x_02_data, x_03_data, x_04_data, x_05_data, 
                  x_06_data, x_07_data, x_08_data) as train_model:
    
    # run MCMC
    varying_intercept_slope_noncentered_trace = pm.sample(draws = draws, tune = tune, chains = chains, 
                                cores = cores, target_accept = 0.95, discard_tuned_samples = True) # very slow, but it works.


###############################################################################
## 9. PICKLING THE TRACE BY JOBLIB 

# save the trace (alternative method 1)
pm.save_trace(trace = varying_intercept_slope_noncentered_trace, 
              directory = os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_7]),
                                           overwrite = True)
  

# shows execution time
print( time.time() - start_time, "seconds")
    
    
