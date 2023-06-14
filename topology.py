import numpy as np
import itertools


def four_node_topology():
    #Empty list to store all topology combinations
    topology_all = []

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                for h in itertools.product([-1,0,1], repeat = 1*4):

                    l1 = [0, -1, 0, i]
                    l2 = [1, 0, -1, j]
                    l3 = [-1, -1, 1, k]
                    l4 = list(h)

                    x = np.concatenate((l1, l2, l3, l4))

                    topology_all.append(x.tolist())

    
    topology_all.remove([0, -1, 0, 0, 1, 0, -1, 0, -1, -1, 1, 0, 0, 0, 0, -1])
    topology_all.remove([0, -1, 0, 0, 1, 0, -1, 0, -1, -1, 1, 0, 0, 0, 0, 0])
    topology_all.remove([0, -1, 0, 0, 1, 0, -1, 0, -1, -1, 1, 0, 0, 0, 0, 1])

    ln = len(topology_all)

    # create topology dictionary
    n = np.arange(1,ln+1,1)
    n = list(map(str, n))  # make keys strings rather than int64 as json file does not allow int64 as file name

    # n = list(np.arange(1,ln+1,1))

    topology_dic = dict(zip(n, topology_all))

    return (topology_dic)


# Filtered out topologies ensuring only networks where node 4 (D) interacts
# exactly with two other nodes remain; 576 such networks (can confirm with len(four_node_filter(topology_dic)))
def four_node_filter(topology_dic):
    filtered_topologies_dic = {}
    
    # Func below checks whether node D interacts with another node; y_i and y_j
    # are diagonally opposite adjacency matrix elements, e.g. y_3 and y_12
    def interaction_checker(y_i, y_j): 
        if y_i == 0 and y_j == 0:
            return 0 # nodes are non-interacting
        else:
            return 1 # nodes are interacting
                
    for key in topology_dic.keys(): 
        y_3, y_12 = topology_dic[key][3], topology_dic[key][12]
        y_7, y_13 = topology_dic[key][7], topology_dic[key][13]
        y_11, y_14 = topology_dic[key][11], topology_dic[key][14]
        
        # if statement below checks for precisely 2 interactions between node D
        # and the other nodes
        if interaction_checker(y_3, y_12) + interaction_checker(y_7, y_13) + interaction_checker(y_11, y_14) == 2: 
            filtered_topologies_dic[key]=topology_dic[key]
    
    return filtered_topologies_dic





# a = four_node_topology()
# print(four_node_filter(a))



# print(four_node_topology())

# generate 4_node topologies as lists and save as dictionary

# topology_dic = topology.four_node_topology()

# print(topology_dic)
# print(topology_dic[1])























# # save topologies as lists

# topology_all = []

# for i in itertools.product([-1, 0, 1], repeat = 2*2):
#   x = list(i)

#   #x = np.reshape(i, (2, 2))
#   topology_all.append(x)

# # 81 possibilities
# #len(topology_all)

# # create topology dictionary
# n = list(np.arange(1,82,1))

# topology_dic = dict(zip(n, topology_all))
# topology_dic