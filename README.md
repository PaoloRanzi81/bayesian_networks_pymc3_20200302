# bayesian_networks_pymc3_20200302
Bayesian network exact linear regression by using polynomials for each edge by PyMC3

PROJECT NAME: 'bayesian_networks_exact_regression_20200202'
AUTHOR: Paolo Ranzi
README FILE VERSION (LAST UPDATE): 20200301


Python 3.6.7 has been used. For each step the specific Python script has been mentioned, accordingly. At the begginning of each script we have to make sure of setting custom-made parameters/pieces of information: 
- import Python libraries (if you do not have a specific library you have to manually install it by using PIP within your virtual enviroment);  
- setting paths and keywords according to your storage location;
- set parameters (e.g. input file name, output file name etc.); 
- all scripts have been optimized for using them by parallel computing. Please set number of CPUs by searching for 'cpu_count()' within each script according to your available resources; 


STEPS: 

01. LEARNING STEP BY EXACT POLYNOMIAL LINEAR REGRESSION FOR EACH EDGE:
(it computes Bayesian + MCMC exact linear regression by PyMC3 for each edge twice. It is a completely data-driven approach, without the introduction
of neither expert-knowledge nor conditional probability table): 
SCRIPT NAME: '01_analysis_20200202.py'
INPUT: .csv file; 
OUTPUT:  pickled bayesian model ( .joblib); pickled MCMC traces (to be used later for diagnositics and prediction);

02. PLOTTING THE DIRECTED ACYCLIC GRAPH (DAG) GRAPH: 
(it computes DAG from the results of single edge linear regressions. In a nutshell, it takes the slope for the interaction between 2 nodes and it plots
it after selecting the strongest causal relationships by thresholding):
SCRIPT NAME: '02_analysis_graph_20200202.py'
INPUT: '01_analysis_20200202.py' + pickled bayesian model ( .joblib);
OUTPUT: DAG in a .pdf format; 










