# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 

def crossval_strat(X, Y, n_iterations, iteration):
    label_set = np.unique(Y)


    label_indices = []
    for label in label_set:
        label_indices.append(np.array([i for i in range(len(Y)) if Y[i]==label]))
    #print(f"label_ind = {label_indices}")

    label_indices_test = []
    for indices in label_indices:
        #print(f"indices = {indices}")
        label_indices_test.append(np.array(indices[iteration*(len(indices) // n_iterations ): (iteration+1)*(len(indices) // n_iterations) ]))
        #print(f"machin {np.array(indices[iteration*(len(indices) // n_iterations ): (iteration+1)*(len(indices) // n_iterations) ])}")

    Xtest = np.array(X[label_indices_test[0]])
    Ytest = np.array(Y[label_indices_test[0]])
    for i in range(1,len(label_indices_test)):
        Xtest = np.concatenate((Xtest, X[label_indices_test[i]]))
        Ytest = np.concatenate((Ytest, Y[label_indices_test[i]]))

    


 


    # index_pos = [ i for i in range(len(Y)) if Y[i]==label_set[0]]
    # index_neg = [ i for i in range(len(Y)) if Y[i]==label_set[1]]

    # index_pos_test = index_pos[iteration*(len(index_pos) // n_iterations ): (iteration+1)*(len(index_pos) // n_iterations) ]
    # index_neg_test = index_neg[iteration*(len(index_neg) // n_iterations ): (iteration+1)*(len(index_neg) // n_iterations) ]
    
    

    # Xtest = np.concatenate(( X[index_neg_test] , X[index_pos_test]  ))
    # Ytest =  np.concatenate(( Y[index_neg_test] , Y[index_pos_test] ))
    #print(label_indices_test)
    indices_test = [elt for l in label_indices_test for elt in l] #flatten
    #print(indices_test)

    index_app = [i for i in range(len(Y)) if (i not in indices_test) ]

    

    #index_app = [i for i in range(len(Y)) if ( (i not in index_pos_test) and (i not in index_neg_test) )]

    Xapp =  X[index_app]
    Yapp =  Y[index_app]

    return Xapp, Yapp, Xtest, Ytest

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return (np.mean(L),np.std(L))

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    for i in range(nb_iter):
        Xapp,Yapp,Xtest,Ytest = crossval_strat(X,Y,nb_iter,i)
        classifier = copy.deepcopy(C)
        classifier.train(Xapp, Yapp)
        perf.append(classifier.accuracy(Xtest,Ytest))
        
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

