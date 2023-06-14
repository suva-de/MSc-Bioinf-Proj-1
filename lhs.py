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


def loguniform(low=-3, high=3, size=None): 
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
    parameters_lhs = ['k_AA','k_BA','k_CA','k_DA',
                    'k_AB','k_BB','k_CB','k_DB',
                    'k_AC','k_BC','k_CC','k_DC',
                    'k_AD','k_BD','k_CD','k_DD',
                    'mu_A','mu_B','mu_C','mu_D',
                    'diffusionConstants_B']
    

    k_AA_range = (0.1, 100)
    k_BA_range = (0.1, 100)
    k_CA_range = (0.1, 100)
    k_DA_range = (0.1, 100)

    k_AB_range = (0.1, 100)
    k_BB_range = (0.1, 100)
    k_CB_range = (0.1, 100)
    k_DB_range = (0.1, 100)

    k_AC_range = (0.1, 100)
    k_BC_range = (0.1, 100)
    k_CC_range = (0.1, 100)
    k_DC_range = (0.1, 100)

    k_AD_range = (0.1, 100)
    k_BD_range = (0.1, 100)
    k_CD_range = (0.1, 100)
    k_DD_range = (0.1, 100)

    mu_A_range = (0.01, 1)
    mu_B_range = (0.01, 1)
    mu_C_range = (0.01, 1)
    mu_D_range = (0.01, 1)

    diffusionConstants_B_range = (1e-3,1e3)

    parameter_range_list = [k_AA_range,k_BA_range,k_CA_range,k_DA_range,
                            k_AB_range,k_BB_range,k_CB_range,k_DB_range,
                            k_AC_range,k_BC_range,k_CC_range,k_DC_range,
                            k_AD_range,k_BD_range,k_CD_range,k_DD_range,
                            mu_A_range,mu_B_range,mu_C_range,mu_D_range,
                            diffusionConstants_B_range]
    
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

    k_AA = lhs(k_AA_distribution,n_param_sets)
    k_BA = lhs(k_BA_distribution,n_param_sets)
    k_CA = lhs(k_CA_distribution,n_param_sets)
    k_DA = lhs(k_DA_distribution,n_param_sets)
    
    k_AB = lhs(k_AB_distribution,n_param_sets)
    k_BB = lhs(k_BB_distribution,n_param_sets)
    k_CB = lhs(k_CB_distribution,n_param_sets)
    k_DB = lhs(k_DB_distribution,n_param_sets)
    
    k_AC = lhs(k_AC_distribution,n_param_sets)
    k_BC = lhs(k_BC_distribution,n_param_sets)
    k_CC = lhs(k_CC_distribution,n_param_sets)
    k_DC = lhs(k_DC_distribution,n_param_sets)
    
    k_AD = lhs(k_AD_distribution,n_param_sets)
    k_BD = lhs(k_BD_distribution,n_param_sets)
    k_CD = lhs(k_CD_distribution,n_param_sets)
    k_DD = lhs(k_DD_distribution,n_param_sets)
    
    mu_A = lhs(mu_A_distribution,n_param_sets)
    mu_B = lhs(mu_B_distribution,n_param_sets)
    mu_C = lhs(mu_C_distribution,n_param_sets)
    mu_D = lhs(mu_D_distribution,n_param_sets)

    diffusionConstants_B = lhs(diffusionConstants_B_distribution,n_param_sets)

# #_________________________________________________________________________________________________________#

        #Define index column to identify parameter sets. 
    index = np.arange(1, n_param_sets + 1, dtype=np.int).reshape(n_param_sets, 1) 
    
    #Define dataframe column names. Then concatenate LHS results.
    parameternames = ['index',
                    'k_AA','k_BA','k_CA','k_DA',
                    'k_AB','k_BB','k_CB','k_DB',
                    'k_AC','k_BC','k_CC','k_DC',
                    'k_AD','k_BD','k_CD','k_DD',
                    'mu_A','mu_B','mu_C','mu_D',
                    'diffusionConstants_B']

    points = np.concatenate((index,
                             k_AA,k_BA,k_CA,k_DA,
                             k_AB,k_BB,k_CB,k_DB,
                             k_AC,k_BC,k_CC,k_DC,
                             k_AD,k_BD,k_CD,k_DD,
                             mu_A,mu_B,mu_C,mu_D,
                             diffusionConstants_B), 1)
    
    df = pd.DataFrame(data=points, columns=parameternames)
    df['index'] = df['index'].astype(int)
    df = df.set_index('index')

    # ### Creation of Hill coefficients dataframe which will be joined to df
    # hill_coeff_labels = ['n_AA','n_BA','n_CA','n_DA',
    #                      'n_AB','n_BB','n_CB','n_DB',
    #                      'n_AC','n_BC','n_CC','n_DC',
    #                      'n_AD','n_BD','n_CD','n_DD'] # Labels will be set as keys in hill_coeff_dict by for loop below
    # hill_coeff_dict = {} # Used to create hill_coeff_df after for loop
    
    # for label in hill_coeff_labels:
    #     hill_coeffs = np.random.randint(2, 5, size=n_param_sets+1) # Drawing n_param_sets + 1 Hill coefficients from uniform distribution over values 2,3,4;
    #     hill_coeff_dict[label] = hill_coeffs                       # this will result in index values from 0 to n_param_sets in hill_coeff_df below. Since
    #                                                                 # there is no 0 index in df, the join between df and hill_coeff_df will be over indices  
    #                                                                 # 1 to n_param_sets. Each Hill coeff column will end up with n_param_sets random values.
    # hill_coeff_df = pd.DataFrame(hill_coeff_dict)
    # df = df.join(hill_coeff_df)

    # df = df.set_index('index')




    
    ##Uncomment pickle.dump if new resampling is to be saved.

    # pickle.dump(df, open('df.pkl', 'wb'))
    # print (df)

#     return df #number of columns is the number of parameters. number of rows is the number of parameter sets / samples






#     # parameterfile_creator_function(12)

# #Call function and dump(save) output to pickle dump for later use 
# n_param_sets = 10000 
# df = parameter(n_param_sets)
# print(df)


#Plot out sampling

# fig = plt.figure(figsize=(10,4))
# ax = fig.add_subplot(122, projection='3d')

# ax.scatter(df['V_A'],df['V_B'],df['k_AA'] )

# ax3 = fig.add_subplot(121)

# plt.scatter(df['V_A'], df['V_B'])
# plt.show()

