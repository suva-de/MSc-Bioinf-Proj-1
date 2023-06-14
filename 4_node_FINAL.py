
import json
import pickle
import numpy as np
from scipy import optimize
import sympy as sp
from sympy import *
from sympy import diff
from numpy import linalg as LA
import multiprocessing as mp

import topology

#___________________________________________________________________________________________________________________________________________________#

df = open('df.pkl', 'rb')
parameters = pickle.load(df)
# add parameters from the paper to the dataframe as a row
# parameters.loc[10001] = [100, 100, 100, 100, 3.1623, 3.1623, 0.00000001,0.00000001, 100, 0.00000001, 3.1623,0.00000001,
#                          100, 3.1623, 0.1,10**50,0.00000001,0.00000001,0.00000001,0.00000001, 0.1, 0.1, 0.1,0.1, 0.01, 0.01, 0.01,0.01, 0.01, 2, 2, 2,0, 2, 2, 2,0, 2, 2, 2, 4, 0, 0, 0, 0]

# (V_A,V_B,V_C,V_D,k_AA,k_BA,k_CA,k_DA,k_AB,k_BB,k_CB,k_DB,k_AC,k_BC,k_CC,k_DC,k_AD,k_BD,k_CD,k_DD,b_A,b_B,b_C,b_D,mu_A,mu_B,mu_C,mu_D,diffusionConstants_B, n_AA,n_BA,n_CA,n_DA,n_AB,n_BB,n_CB,n_DB,n_AC,n_BC,n_CC,n_DC,n_AD,n_BD,n_CD,n_DD)
#k_DC = 10**50 and n_DC = 4; this is so that (d/k_DC)**(n_DC) in reaction term fc(a, b, c, d) below is negligibly small 

# print (parameters.loc[9])
# print (parameters)
# ___________________________________________________________________________________________________________________________________________________

# load LHS for initial conditions

init_con_df = open('init_con_df.pkl', 'rb')
initial_con = pickle.load(init_con_df)
initial_con = initial_con.to_numpy()
# print(initial_con)
# ___________________________________________________________________________________________________________________________________________________

# generate 4_node topologies as lists and save as dictionary

topology_dic = topology.four_node_topology()
topology_dic = [items for items in topology_dic.items()]

# print(topology_dic)
# print(len(topology_dic))
# print(topology_dic[1])
# ___________________________________________________________________________________________________________________________________________________


def assign(topology, params, top_dispersion):
    '''
    Args:
        topology: a key:value pair from the topology dictionary (topology_dic.items())
                    e.g.
                        topology_dic= {3954:[1,-1,0,1,0,-1,-1,-1,1]}
                        dic_items = [items for items in topology_dic.items()]
                        dic_items[0] >>> (3954, [1, -1, 0, 1, 0, -1, -1, -1, 1])
                        dic_items[0][0] >>> 3954
                        dic_items[0][1] >>> [1, -1, 0, 1, 0, -1, -1, -1, 1]

        params: a row from the LHS dataframe
                i.e., [df.loc[i] for i in range(1,5)]
        top_dispersion: 5000

    Output: 
        turing_output: a dictionary with Turing output results
                       e.g. {key:["Turing_status", parameter set number, score or classification]}
    '''

    key = topology[0]  # assign topology name (e.g. 3954)
    # assign adjacency matrix (list) (e.g. [1, -1, 0, 1, 0, -1, -1, -1, 1])
    mat_elems = topology[1]
    n_species = 4

    parameter_set = params.name  # assign parameter_set number

    # assign values to each parameter
    k_AA = params.at['k_AA']
    k_BA = params.at['k_BA']
    k_CA = params.at['k_CA']
    k_DA = params.at['k_DA']
    
    k_AB = params.at['k_AB']
    k_BB = params.at['k_BB']
    k_CB = params.at['k_CB']
    k_DB = params.at['k_DB']
    
    k_AC = params.at['k_AC']
    k_BC = params.at['k_BC']
    k_CC = params.at['k_CC']
    k_DC = params.at['k_DC']
    
    k_AD = params.at['k_AD']
    k_BD = params.at['k_BD']
    k_CD = params.at['k_CD']
    k_DD = params.at['k_DD']

    mu_A, mu_B, mu_C, mu_D = params.at['mu_A'], params.at['mu_B'], params.at['mu_C'], params.at['mu_D']
    d_B = params.at['diffusionConstants_B']
    
   
    # Assignment of y_i, which enables mapping from adjacency matrix/topology to reaction terms below
    y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8 = mat_elems[0], mat_elems[1], mat_elems[2], mat_elems[3], mat_elems[4], mat_elems[5], mat_elems[6], mat_elems[7], mat_elems[8]
    y_9, y_10, y_11, y_12, y_13, y_14, y_15 = mat_elems[9], mat_elems[10], mat_elems[11], mat_elems[12], mat_elems[13], mat_elems[14], mat_elems[15] 
                                              
    # Define each equation
    a, b, c, d = symbols('a b c d')

    def fa(x):
        return 100 * (1/(1+(x[0]/k_AA)**(-y_0*2)))**np.abs(y_0) * (1/(1+(x[1]/k_BA)**(-y_1*2)))**np.abs(y_1) * (1/(1+(x[2]/k_CA)**(-y_2*2)))**np.abs(y_2) * (1/(1+(x[3]/k_DA)**(-y_3*2)))**np.abs(y_3) + 0.1 - mu_A*x[0]

    def fb(x):
        return 100 * (1/(1+(x[0]/k_AB)**(-y_4*2)))**np.abs(y_4) * (1/(1+(x[1]/k_BB)**(-y_5*2)))**np.abs(y_5) * (1/(1+(x[2]/k_CB)**(-y_6*2)))**np.abs(y_6) * (1/(1+(x[3]/k_DB)**(-y_7*2)))**np.abs(y_7) + 0.1 - mu_B*x[1]

    def fc(x):
        return 100 * (1/(1+(x[0]/k_AC)**(-y_8*2)))**np.abs(y_8) * (1/(1+(x[1]/k_BC)**(-y_9*2)))**np.abs(y_9) * (1/(1+(x[2]/k_CC)**(-y_10*2)))**np.abs(y_10) * (1/(1+(x[3]/k_DC)**(-y_11*2)))**np.abs(y_11) + 0.1 - mu_C*x[2]

    def fd(x):
        return 100 * (1/(1+(x[0]/k_AD)**(-y_12*2)))**np.abs(y_12) * (1/(1+(x[1]/k_BD)**(-y_13*2)))**np.abs(y_13) * (1/(1+(x[2]/k_CD)**(-y_14*2)))**np.abs(y_14) * (1/(1+(x[3]/k_DD)**(-y_15*2)))**np.abs(y_15) + 0.1 - mu_D*x[3]

    def f_4Node(x):
        return [fa(x), fb(x), fc(x), fd(x)]
    
    X = [a, b, c, d]
    equation = sp.Matrix([fa(X),fb(X),fc(X),fd(X)])
    # print(equation)
    vars = sp.Matrix([a, b, c, d])
    Jacobian = equation.jacobian(vars)
    # print(Jacobian)

    # Searching for roots over a grid
    search_range = np.linspace(0, 2000, 10)
    roots = []

    for guess in initial_con:

        root = optimize.root(f_4Node, guess, method='hybr')
        res = np.all(root.x >= 0)
        val = np.around(root.x, decimals=3)
        # Filter out negative concentrations, fake roots, and repeats
        if res == True and root.success == True and val.tolist() not in roots:
            roots.append(val.tolist())
    print(roots)

    
    counter_t1 = 0

    for rt_index in range(len(roots)):
        u = roots[rt_index]

        # jacobian without diffusion
        jac = np.array(Jacobian.subs([(a, u[0]), (b, u[1]), (c, u[2]), (d, u[3])]), dtype=float)
        # print(jac)

        # Compute eigenval and eigen vector of Jacobian Matrix
        eig_val, eig_vector = LA.eig(jac)
        # print (eig_val.real)

        if all(eig_val.real < 0):    # filter for stability without diffusion
            wvn_list = np.array(list(np.arange(0, top_dispersion+1)))*np.pi/100
            count = 0
            eigenvalues = np.zeros((len(wvn_list), n_species), dtype=complex)
            turing_eigen = []
            max_eigen = []

            for wvn in wvn_list:
                # add diffusion
                diffusion_terms = np.array([[wvn**2, 0, 0, 0],
                                           [0, d_B*(wvn**2), 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]])
                
                # Jacobian with Diffusion (only node A and B are diffusing)
                jac_with_diffusion = jac - diffusion_terms
                eigenval, eigenvec = LA.eig(jac_with_diffusion)

                # np.argsort -- sort eigenvalues so the one with the instability is at position -1.
                idx = np.argsort(eigenval)
                eigenval = eigenval[idx]  # orders eigenvalues for each k.

                eigenvalues[count] = eigenval
                count += 1

                # Capture all +ve eigen value for all wavenumbers in list (turing_eigen)
                for r in eigenval.real:
                    if r > 0:
                        turing_eigen.append(r)

                e = max(eigenval.real)
                max_eigen.append(e)
            # print (max_eigen)
            # print (turing_eigen)

            if max(max_eigen) > 0:  # filter for Turing instability with diffusion
                max_value = max(max_eigen)
                max_index = max_eigen.index(max_value)

                # Turing I condition
                if max_index < top_dispersion:
                    counter_t1 += 1
                    if len(roots)==1:
                        turing_output = {key:["Turing_1", int(parameter_set), 1]}
                        # print(f"Turing II for topology {key} df {parameter_set}")

                # Turing II condition 
                else:
                    if len(roots)==1:
                        turing_output = {key:["Turing_2", int(parameter_set), 1]}
                        # print(f"Turing II for topology {key} df {parameter_set}")
                
            else: # no Turing pattern
                if len(roots)==1:
                    turing_output = {key:["Turing_0", int(parameter_set), 1]}
                    # print(f"No Turing for topology {key} df {parameter_set}")
        
        else: # single steady state but unstable without diffusion
            turing_output = {key:["No_Turing", int(parameter_set), "unstable without diffusion"]}



    if len(roots) > 1:
        if counter_t1 != 0:
            weighted_score = (counter_t1)/(len(roots))
            turing_output = {key:["Turing_1", int(parameter_set), weighted_score]}
            # print(f"Turing I with score {weighted_score} for topology {key} df {parameter_set}. Turing eigens are {turing_eigen}  ")

        else:
            turing_output = {key:["No_Turing", int(parameter_set), "multi-steady"]}
            # output_no_turing.append([{key:f"No Turing: param set {parameter_set}"}])
            # print(f"No Turing for topology {key} df {parameter_set}")

    print(turing_output)

    return turing_output


# # Test
# topo = topology_dic[0]
# params = parameters.loc[1]

# assign(topo, params, 5000)


# asynchro
if __name__ == '__main__':

    for h in range(len(topology_dic)+1):
    # for h in range(0,301):

        topo = topology_dic[h]
        key = topo[0]  # topology name (e.g. 3954)

        pool = mp.Pool(mp.cpu_count())
        turing_results = []

        def collect_result(result):
            global turing_results
            turing_results.append(result)

        for i in range(1,len(parameters)+1):
        # for i in range(1,9):

            pool.apply_async(assign, args=(topo, parameters.loc[i], 5000), callback = collect_result)

        pool.close()
        pool.join()

        def sortFunc(e):
            return e[key][1]

        turing_results.sort(key=sortFunc)
        results_final = [r for r in turing_results]

        print(results_final)

        with open(f"turing_output_topo_{key}.json", "w") as save_turing:
            json.dump(results_final, save_turing)


