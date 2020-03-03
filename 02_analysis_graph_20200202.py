"""
TITLE: "Plotting DAG from Bayesian Networks exact linear regression"
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
import datetime
import numpy as np
import pandas as pd
from copy import deepcopy
import pymc3 as pm
import joblib
from multiprocessing import cpu_count
#from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import graphviz as gz



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
output_file_name_2 = ('012_analysis_bayesian_model.joblib') 
output_file_name_3 = ('012_analysis_trace_model.joblib') 
output_file_name_4 = ('012_analysis_X_test_oob.csv') 
output_file_name_7 = ('012_analysis_saved_trace')
output_file_name_8 = ('mask_edges_to_nodes.csv')
output_file_name_9 = ('012_graph.svg')


# start clocking time
start_time = time.time()


###############################################################################
# 4. LOAD PREVIOUSLY TRAINED MODEL AND HOLD-OUT DATA-SET
## load hold-out-set
## loading the .csv file 
X_test_oob_1 = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         output_file_name_4]), header = 0)
       
# load objects 
varying_intercept_slope_noncentered = joblib.load(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2])) 

varying_intercept_slope_noncentered_trace = pm.load_trace(directory = os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_7]), 
                                                          model = varying_intercept_slope_noncentered)

# deepcopy 
X_test_oob = X_test_oob_1.copy()  
X_test_oob_1 = None

## standardize hold-out-test set
## standardize necessary step, otherwise PyMC3 throws errors)
ss_train = StandardScaler()
X_test_oob.loc[:, :] = ss_train.fit_transform(X_test_oob.loc[:, :])   


###############################################################################
## LOAD OUT-OF-BAG SET

# Build hold-out-set (Pandas Series)
# set shared theano variables
x_01_data = X_test_oob.loc[:, 'var_01'].to_numpy()
x_02_data = X_test_oob.loc[:, 'var_02'].to_numpy()
x_03_data = X_test_oob.loc[:, 'var_03'].to_numpy()
x_04_data = X_test_oob.loc[:, 'var_04'].to_numpy()
x_05_data = X_test_oob.loc[:, 'var_05'].to_numpy()
x_06_data = X_test_oob.loc[:, 'var_06'].to_numpy()
x_07_data = X_test_oob.loc[:, 'var_07'].to_numpy()
x_08_data = X_test_oob.loc[:, 'var_08'].to_numpy()


###############################################################################
## SETTING MODEL (e.g. VARYING_INTERCEPT_AND_SLOPE (NON-CENTERED))

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
       
            
    return varying_intercept_slope_noncentered



###############################################################################
## PREDICTING 

# predicting new values from the posterior distribution of the previously trained model
with model_factory(x_01_data, x_02_data, x_03_data, x_04_data, x_05_data, 
                  x_06_data, x_07_data, x_08_data) as test_model:
   trace_df_1 = pm.trace_to_dataframe(varying_intercept_slope_noncentered_trace, include_transformed=True)

   
#trace_df_1.columns
   

###############################################################################
## DATA TRANSFORMATION FROM PYMC3 OUTPUT TO GRAPHVIZ INPUT 
# summary statistics
summary_statistics_tmp = pm.summary(varying_intercept_slope_noncentered_trace)
summary_statistics = summary_statistics_tmp.reset_index(drop = False, inplace = False)

# rename column
summary_statistics.rename(columns={'index': 'single_edge'}, inplace = True)

## check traces' names
#trace_columns = pd.DataFrame(trace_df_1.columns)

# convert list of PyMC3 model's variables into a list of strings
betas_string_tmp = summary_statistics.loc[:, 'single_edge']

# select strings starting with "beta..."
mask = betas_string_tmp.str.startswith('beta')
betas_df = summary_statistics.loc[mask, : ]

# combine results from two nodes but belonging to the same edge
# WARNING 1: this step is a quick-and-dirty workaround since I am not proficient
# with regular expressions. In the future, such a step will be substituted;
# WARNING 2: put 'mask_edges_to_nodes.csv' into BASE_DIR_OUTPUT;
mask_converged_edges = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         output_file_name_8]), header = 0)
    
# merge columns
betas_df_merged_tmp_01 = pd.merge(betas_df, mask_converged_edges, how = 'inner', 
                          on = 'single_edge')

# dictionary for replacing nodes's name to specific variables' names
name_dictionary = {'node_01' : 'var_01', 'node_02' : 'var_02', 'node_03' : 'var_03',
                    'node_04' : 'var_04', 'node_05' : 'var_05', 'node_06' : 'var_06', 
                    'node_07' : 'var_07', 'node_08' : 'var_08'}

# convert dictionary to DataFrame
name_df = pd.DataFrame.from_dict(name_dictionary, orient = 'index', columns = ['specific'])

# reset index
name_df.reset_index(drop = False, inplace = True)

# rename column
name_df.rename(columns = {'index' : 'general'}, inplace = True)

# double single nodes columns (to be used later for replacing strings)
doubled_tail = betas_df_merged_tmp_01.loc[: , 'arrow_tail'].rename('var_tail')
doubled_head = betas_df_merged_tmp_01.loc[: , 'arrow_head'].rename('var_head')

# concatenate
doubled_concat = pd.concat([doubled_tail, doubled_head], axis = 1)

# replace strings
for iii in range(0, len(name_dictionary)):  
    doubled_concat.replace(name_df.loc[iii, 'general'], 
                     name_df.loc[iii, 'specific'], inplace = True) 
    
# concatenate
betas_df_merged_tmp_02 = pd.concat([betas_df_merged_tmp_01, doubled_concat], axis = 1)

# negative 
negative = betas_df_merged_tmp_02.loc[:, 'mean'].lt(0).rename('negative')

# positive 
positive = betas_df_merged_tmp_02.loc[:, 'mean'].gt(0).rename('positive')

# concatenate
betas_df_merged_tmp_03 = pd.concat([betas_df_merged_tmp_02, positive, negative], axis = 1)


###############################################################################
## compute 0.95 quantile (not standardized data):
threshold_first_quantile = betas_df_merged_tmp_03.loc[betas_df_merged_tmp_03.loc[:, 'negative'],
                                                   'mean'].quantile(q = 0.15, interpolation='lower')
threshold_second_quantile = betas_df_merged_tmp_03.loc[betas_df_merged_tmp_03.loc[:, 'positive'], 
                                                    'mean'].quantile(q = 0.85, interpolation='higher')

# subset by threshold 
mask_subset_greater_than = betas_df_merged_tmp_03.loc[:, 'mean'].gt(threshold_second_quantile) 
mask_subset_smaller_than = betas_df_merged_tmp_03.loc[:, 'mean'].lt(threshold_first_quantile)
mask_subset_concat = pd.concat([mask_subset_greater_than, mask_subset_smaller_than], axis = 1)
mask_subset = mask_subset_concat.sum(axis = 1).astype('bool')
subset = betas_df_merged_tmp_03.loc[mask_subset, :].reset_index(drop = True)

# re-scale absolute value of the slope in order to build a standardize arrow's thickness measure
from sklearn.preprocessing import minmax_scale
absolute_values_tmp = subset.loc[:, 'mean'].abs()
min_max = pd.Series(minmax_scale(absolute_values_tmp)).rename('re_scaled')
subset_rescaled = pd.concat([subset, min_max], axis = 1)


###############################################################################
## PLOTTING FLOW DIAGRAM OF CAUSAL RELATIONSHIPS BY GRAPHVIZ

# initialize a Directed Graph
dot = gz.Digraph(comment='8_vars')    

# assign specific nodes' name
for count in range(0, subset_rescaled.shape[0]):
    if subset_rescaled.loc[count, 'positive'] == True:
       dot.edge(str(subset_rescaled.loc[count, 'var_head']), str(subset_rescaled.loc[count, 'var_tail']),
                penwidth = str(subset_rescaled.loc[count, 're_scaled']*10)) # 10 is the scaling factor for arrow's thickness
    else:
       dot.edge(str(subset_rescaled.loc[count, 'var_tail']), str(subset_rescaled.loc[count, 'var_head']),
                penwidth = str(subset_rescaled.loc[count, 're_scaled']*10)) # 10 is the scaling factor for arrow's thickness
       
#print(dot.source)    
dot.view()

# save graphviz as PDF
dot.render(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_9]), view=True) 



"""
###############################################################################
## 10. MCMC TRACE DIAGNOSTICS [to be done only once for calibrating the bayesian model]  


# see graph for model
import graphviz
pm.model_to_graphviz(varying_intercept_slope_noncentered)

# too RAM damanding
data = az.convert_to_dataset(varying_intercept_slope_noncentered_trace)

## show traces
pm.traceplot(varying_intercept_slope_noncentered_trace)  

#az.plot_trace(glm_model_trace, compact=True)
az.plot_trace(varying_intercept_slope_noncentered_trace[:3000], var_names = "υ", divergences = "bottom")
az.plot_trace(varying_intercept_slope_noncentered_trace, var_names = "beta_node_01_node_02", divergences = "bottom")

# save figure 
date = str(datetime.datetime.now()) 

# to get the current figure...       
fig = plt.gcf()

# save figure
fig.savefig(os.path.sep.join([BASE_DIR_OUTPUT, date[0:10] 
+ "_" + date[11:12] + "_" + date[14:15] + date[17:22] + ".svg"])) 

# close pic in order to avoid overwriting with previous pics
fig.clf()    


## show posterior    
pm.plot_posterior(varying_intercept_slope_noncentered_trace)  
# save figure 
date = str(datetime.datetime.now()) 

# to get the current figure...       
fig = plt.gcf()

# save figure
fig.savefig(os.path.sep.join([BASE_DIR_OUTPUT, date[0:10] 
+ "_" + date[11:12] + "_" + date[14:15] + date[17:22] + ".svg"])) 

# close pic in order to avoid overwriting with previous pics
fig.clf() 

# forest plot
forestplot(varying_intercept_slope_noncentered_trace, varnames=['b']);


## summary statistics
summary_statistics_trace_012 = pm.summary(varying_intercept_slope_noncentered_trace)
"""


# shows execution time
print( time.time() - start_time, "seconds")




