import pickle
from re import A
from argon2 import Parameters
import pandas as pd
import numpy as np
from datetime import date
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

#LOGUNIFORM function returns a loguniform distribution from 'low' to 'high' of a certain 'size'. 
#In the loguniform distribution, a positive random variable X is log-uniformly distributed if the logarithm of X is uniform distributed. 
#Also called the reciprocal distribution

#function (1)


def loguniform(low=-3, high=4, size=None): 
    return (10) ** (np.random.uniform(low, high, size))

#_________________________________________________________________________________________________________#


#function (2)
def lhs(data, nsample): 
    #data contains the information on the distribution. 
    #nsample is the number of samples desired (x). 
    m, nvar = len(data), 1
    #m = len(data) and  nvar = 1

    ran = np.random.uniform(size=(nsample, nvar))
    s = np.zeros((nsample, nvar))
    for j in range(0, nvar):
        idx = np.random.permutation(nsample) + 1
        P = ((idx - ran[:, j]) / nsample) * 100
        s[:, j] = np.percentile(data[:, j], P)
    return s

#_________________________________________________________________________________________________________#


#function (3)

def parameter(n_param_sets): #number of parameter sets desired in the df (number of rows)
    
    #First, an example distribution is created (loguniform or uniform).
    loguniformdist = loguniform(size=100) 
    
    #Parameters and their possible ranges
    parameters_lhs = ['a','b','c','d']
    
    a_range = (0, 10000)
    b_range = (0, 10000)
    c_range = (0, 10000)
    d_range = (0, 10000)

    parameter_range_list = [a_range,b_range,c_range,d_range]
    
#_________________________________________________________________________________________________________#

    #1. Adapt distribution to parameter range parameter_distribution_list = []
    parameter_distribution_list = []
    for parameter_range in parameter_range_list: 
        
        distribution = [x for x in loguniformdist if parameter_range[0] <= x <= parameter_range[1]]
        # this only selects distribution values derived from log_uniform_dist if it is within the ranges of the parameter range, i.e. V_A range of 0.1 to 100. Anything less or above is excluded. 
        # and since the loguniformdist is the distribution we want to sample from, all this does is limits the sampled values to the ranges of the parameters_range.  
        parameter_distribution_list.append(distribution)
    
    # print (parameter_distribution_list) 

#_________________________________________________________________________________________________________#

    #2. Adapt distributions so they are all of the same length
    minimumlenghtdistribution = np.amin([len(x) for x in parameter_distribution_list])
    # print (minimumlengthdistribution)
    for count, parameter_distribution in enumerate(parameter_distribution_list):
        #count counts the number of rows in the distribution list....i.e the sampling number 
        #parameter distribution prints out the values of the sampling 
        # print (parameter_distribution)
        # print (count, parameter_distribution)

        globals()[f'{parameters_lhs[count]}_distribution'] =  np.column_stack( (parameter_distribution[:minimumlenghtdistribution])).transpose()


    #Sample using LHS

    a = lhs(a_distribution,n_param_sets)
    b = lhs(b_distribution,n_param_sets)
    c = lhs(c_distribution,n_param_sets)
    d = lhs(d_distribution,n_param_sets)

# #_________________________________________________________________________________________________________#

        #Define index column to identify parameter sets. 
    index = np.arange(1, n_param_sets + 1, dtype=np.int).reshape(n_param_sets, 1) 
    
    #Define dataframe column names. Then concatenate LHS results.
    parameternames = ['index','a','b','c','d']

    points = np.concatenate((index,a,b,c,d), 1)
    
    df = pd.DataFrame(data=points, columns=parameternames)
    df['index'] = df['index'].astype(int)
    df = df.set_index('index')



    
    ##Uncomment pickle.dump if new resampling is to be saved.

    # pickle.dump(df, open('init_con_df.pkl', 'wb'))
    # print (df)



#     # parameterfile_creator_function(12)

# #Call function and dump(save) output to pickle dump for later use 
# n_param_sets = 100 
# df = parameter(n_param_sets)
# print(df)