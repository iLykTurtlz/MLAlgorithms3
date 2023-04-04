# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2023

# Import de packages externes
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.spatial import distance

# ---------------------------

# ------------------------ A COMPLETER :

# Recopier ici la classe Classifier (complète) du TME 2

# ------------------------ A COMPLETER :
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI : 
        cpt=0
        n = len(desc_set)
        predict_label = []
        for i in range(n):
            predict_label.append(self.predict(desc_set[i]))
        accuracy= np.mean(np.array(predict_label)==np.array(label_set))
        return accuracy
        
        # ------------------------------
        
        # ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON



class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if init:
            self.w = np.zeros(self.input_dimension)
        else:
            self.w = np.asarray([(2*(np.random.uniform(0,1)) - 1)*0.0001 for i in range(input_dimension)])

        self.allw =[self.w.copy()]
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
 
        desc_set2, label_set2 = shuffle(desc_set, label_set)

        for xi,yi in zip(desc_set2, label_set2):
            if self.predict(xi) != yi:
                self.w = self.w + self.learning_rate*yi*xi
                self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        liste_difference=[]
        
        for i in range(nb_max):
            w_avant = self.w.copy()
            self.train_step(desc_set,label_set)
            difference = np.abs(self.w - w_avant)
            change = np.linalg.norm(difference)
            liste_difference.append(change)
            
            if change <= seuil:
                break
                
        return liste_difference
        
    def get_allw(self):
        return self.allw
            
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """

        return np.vdot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """

        return 1 if self.score(x) > 0 else -1
    
# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        #raise NotImplementedError("Please Implement this method")
        self.input_dimension = input_dimension
        self.k = k
        
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        #raise NotImplementedError("Please Implement this method")
        #dist = [np.sqrt((x[0]-self.desc_set[i][0])**2+(x[1]-self.desc_set[i][1])**2) for i in range(len(self.desc_set))]
        dst=[]
        for i in range(len(self.desc_set)):
          dst.append(distance.euclidean(self.desc_set[i],x))
        ordre = np.argsort(dst)
        ordre = ordre[:self.k]
        
        nombre_pos=0
        for i in ordre:
            if self.label_set[i] == +1:
                nombre_pos += 1
        p = nombre_pos/self.k
        return 2*(p-0.5)
        
    
        #return len([])
        
        
        
        '''
        
        # Extraction des exemples de classe -1:
data2_negatifs = data2_desc[data2_label == -1]
# Extraction des exemples de classe +1:
data2_positifs = data2_desc[data2_label == +1]
    '''


    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        #raise NotImplementedError("Please Implement this method")
        if self.score(x) > 0:
            return +1
        else:
            return -1 #on ne considère pas le cas score==0

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        #raise NotImplementedError("Please Implement this method")
        self.desc_set = desc_set
        self.label_set = label_set

