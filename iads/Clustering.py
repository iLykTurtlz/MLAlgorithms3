# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt

# ------------------------ 

def normalisation(frame):
    return pd.DataFrame({col : (frame[col] - mini)/(maxi-mini) for col, mini, maxi in zip(frame.columns, frame.min(axis=0), frame.max(axis=0))})


def dist_euclidienne(v1, v2):
    return np.sqrt(np.sum(((v1 - v2)**2)))


def centroide(dataset):
    return dataset.sum(axis=0) / dataset.shape[0]


def dist_centroides(dataset1, dataset2):
    return dist_euclidienne(centroide(dataset1), centroide(dataset2))


def dist_complete(dataset1, dataset2):      #optimiser???
    max_dist = 0
    for i in range(dataset1.shape[0]):
        for j in range(dataset2.shape[0]):
            if (  current_dist := dist_euclidienne(dataset1.iloc[i], dataset2.iloc[j])  ) > max_dist:
                max_dist = current_dist
    return max_dist


def dist_simple(dataset1, dataset2):
    min_dist = np.inf
    for i in range(dataset1.shape[0]):
        for j in range(dataset2.shape[0]):
            if (  current_dist := dist_euclidienne(dataset1.iloc[i], dataset2.iloc[j])  ) < min_dist:
                min_dist = current_dist
    return min_dist


def dist_average(dataset1, dataset2):
    total_dist = 0
    for i in range(dataset1.shape[0]):
        for j in range(dataset2.shape[0]):
            total_dist += dist_euclidienne(dataset1.iloc[i],dataset2.iloc[j])
    return total_dist / (dataset1.shape[0] + dataset2.shape[0])


def initialise_CHA(DF):
    return {i:[i] for i in range(DF.shape[0])}


def fusionne(DF, P0, linkage="centroid", verbose=False):
    """
    Fusionne les deux clusters de P0 les plus proches selon le linkage demandé parmi {"centroide", "complete", "simple", "average"}
    """
    if linkage == "centroid":       #ici il faut choisir le critère de linkage
        dist = dist_centroides
    elif linkage == "complete":
        dist = dist_complete
    elif linkage == "simple":
        dist = dist_simple
    elif linkage == "average":
        dist = dist_average

    min_distance = np.inf
    removed_key1 = -1
    removed_key2 = -1
    key_list = list(P0.keys())      #apparemment on ne peut pas transformer un objet de type dict_keys en nparray
    for i,key1 in enumerate(key_list):
        for j in range(i+1,len(key_list)):
            key2 = key_list[j]
            if (  current_distance := dist(DF.iloc[P0[key1]], DF.iloc[P0[key2]])  ) < min_distance:
                min_distance = current_distance
                removed_key1 = key1
                removed_key2 = key2
    if verbose:
        print(f"Distance mininimale trouvée entre  [{removed_key1}, {removed_key2}]  =  {min_distance}")
    P1 = {k:v for k,v in P0.items() if k != removed_key1 and k != removed_key2}
    new_key = max(P0.keys()) + 1
    P1[new_key] = [i for i in P0[removed_key1] + P0[removed_key2]]
    return P1, removed_key1, removed_key2, min_distance


#Cette fonction fait tout ce qui est demandé
def clustering_hierarchique(DF, linkage="centroid", verbose=False, dendrogramme=False):
    P = initialise_CHA(DF)
    result = []
    while len(P) > 1:
        next_key = max(P.keys()) + 1        #autre méthode : garder l'ancien P et additionner les longueurs de P[removed_key1] et de P[removed_key2]
        P, removed_key1, removed_key2, min_distance = fusionne(DF, P, linkage=linkage, verbose=verbose)
        result.append([removed_key1, removed_key2, min_distance, len(P[next_key])])
    
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            result, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    
    return result


def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """
    Clustering hiérarchique ascendante.  Cette fonction prend un dataframe ou numpy array et fait le clustering selon
    un critère de linkage.  Elle permet également d'afficher un dendrogramme et un resumé de chaque étape de l'algorithme.

    À rajouter plus tard: critère de distance modifiable.
    """
    clustering_hierarchique(DF, linkage=linkage, verbose=verbose, dendrogramme=dendrogramme)


def CHA_centroid(DF, verbose=False, dendrogramme=False):
    P = initialise_CHA(DF)
    result = []
    while len(P) > 1:
        next_key = max(P.keys()) + 1        #autre méthode : garder l'ancien P et additionner les longueurs de P[removed_key1] et de P[removed_key2]
        P, removed_key1, removed_key2, min_distance = fusionne(DF, P, verbose=verbose)
        result.append([removed_key1, removed_key2, min_distance, len(P[next_key])])
    
    if dendrogramme:
        # Paramètre de la fenêtre d'affichage: 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            result, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    
    return result



#kmoyennes
def inertie_cluster(Ens):
    return np.sum( np.sum(((Ens - centroide(Ens))**2)) )


def init_kmeans(K, Ens):
    ens_array = np.array(Ens)
    indices = np.random.choice(a=Ens.shape[0], size=K, replace=False)
    return ens_array[indices]


def plus_proche(Exe,Centres):
    dist_min = np.inf
    indice_min = -1
    for i in range(Centres.shape[0]):
        if (this_distance := dist_euclidienne(Exe,Centres[i])) < dist_min:
            dist_min = this_distance
            indice_min = i
    return indice_min


def affecte_cluster(Base,Centres):
    result = {j : [] for j in range(len(Centres))}
    for i,row in Base.iterrows():
        result[plus_proche(row, Centres)].append(i)
    return result


def nouveaux_centroides(Base,U):
    return np.asarray([centroide(Base.iloc[v]) for v in U.values()])


def inertie_globale(Base, U):
    return np.sum([inertie_cluster(Base.iloc[v]) for v in U.values()])


def kmoyennes(K, Base, epsilon, iter_max, verbose=False):
    C = init_kmeans(K,Base)                                                          #demander au prof comment il a initialisé J0
    delta = epsilon+1
    J0=0

    i=0
    while i<iter_max and delta > epsilon:
        U = affecte_cluster(Base, C)
        C = nouveaux_centroides(Base, U)
        J1 = inertie_globale(Base, U)
        delta = np.abs(J1-J0)
        J0 = J1
        i += 1
        if verbose:
            print(f'iteration {i} Inertie : {J1:.4f} Difference: {delta:.4f}')
    return C, U


def affiche_resultat(Base,Centres,Affect):
    couleurs = cm.tab20(np.linspace(0, 1, 20))

    assert(len(Affect) <= len(couleurs))    #sinon il faut plus de couleurs

    for x,y in Centres:
        plt.scatter(x,y,color='red', marker = 'x')

    columns = Base.columns
    color_index = 0
    for point_indices in Affect.values():
        this_color = couleurs[color_index]
        xs = Base.iloc[point_indices][columns[0]]
        ys = Base.iloc[point_indices][columns[1]]
        plt.scatter(xs,ys,color=this_color)

        color_index += 1

def min_dist_intercluster(Centres):
    dist_min = np.inf
    for i, centre in enumerate(Centres):
        for j in range(i+1, Centres.shape[0]):
            if (this_distance := dist_euclidienne(centre, Centres[j])) < dist_min:
                dist_min = this_distance
    return dist_min


def max_dist_intracluster(Base, Affect):
    dist_max = -np.inf
    for index, examples in Affect.items():
        for i, example1 in enumerate(examples):
            for j in range(i+1, len(examples)):
                example2 = examples[j]
                if (this_distance := dist_euclidienne(Base.iloc[example1], Base.iloc[example2])) < dist_max:
                    dist_max = this_distance
    return this_distance


def Dunn(Base, Centres, Affect):
    #on veut minimiser
    return max_dist_intracluster(Base, Affect) / min_dist_intercluster(Centres)


def XieBeni(Base, Centres, Affect):
    #on veut minimiser cette valeur
    return inertie_globale(Base, Affect) / min_dist_intercluster(Centres)

def calculate_indices(Base, Centres, Affect):
    """
    Calcule Dunn, XieBeni
    optimisation pour éviter un calcul redondant du dénominateur
    """
    min_dist_inter = min_dist_intercluster(Centres)
    return max_dist_intracluster(Base, Affect) / min_dist_inter, inertie_globale(Base, Affect) / min_dist_inter



