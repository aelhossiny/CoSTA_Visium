##import modules
import cv2
import numpy as np
import pandas as pd
import NaiveDE

##neural net
import torch
import torch.nn.functional as F

import umap.umap_ as umap
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score

from bi_tempered_loss_pytorch import bi_tempered_logistic_loss

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import re
import matplotlib.pyplot as plt
from itertools import combinations

##evalution is the essential function in CoSTA. It performs clustering and generates soft assignment
##
def evaluation(y_pred,cluster_method="Kmeans",num_cluster = 25,n_neighbors=20,min_dist=0.0):
    
    if cluster_method=="Kmeans":
        embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=num_cluster,
                              metric="euclidean").fit_transform(y_pred)
    
        kmeans = KMeans(n_clusters=num_cluster, random_state=1).fit(embedding)
        centroid = kmeans.cluster_centers_.copy()
        y_label = kmeans.labels_.copy()
        y_pseudo=np.zeros((y_pred.shape[0],num_cluster))
    elif cluster_method=="SC":
        embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=num_cluster,
                              metric="euclidean").fit_transform(y_pred)
        clustering = SpectralClustering(n_clusters=num_cluster,
                                        assign_labels="discretize",
                                        random_state=0).fit(embedding)
        y_label = clustering.labels_.copy()
        centroid = pd.DataFrame(embedding.copy())
        centroid['label']=y_label
        centroid = centroid.groupby('label').mean().values
        y_pseudo=np.zeros((y_pred.shape[0],num_cluster))

    else:
        embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=num_cluster,
                              metric="euclidean").fit_transform(y_pred)
        gmm = GaussianMixture(n_components=num_cluster).fit(embedding)
        y_label = gmm.predict(embedding)
        centroid = pd.DataFrame(embedding.copy())
        centroid['label']=y_label
        centroid = centroid.groupby('label').mean().values

        y_pseudo=np.zeros((y_pred.shape[0],num_cluster))
    
    ##t-student distribution kernel soft-assignment,alpha=1
    #for j in range(centroid.shape[0]):
    #    y_pseudo[:,j]=(np.linalg.norm(embedding-centroid[j,:],axis=1)+1)**(-1)
        ##cosine distance
        #y_pseudo[:,j]=((1-cosine_similarity(embedding,centroid[j,:].reshape(1,embedding.shape[1]))+1)**(-1))[:,0]
    #y_pseudo = pd.DataFrame(y_pseudo)
    #y_pseudo2=np.zeros((y_pred.shape[0],centroid.shape[0]))
    #for j in range(centroid.shape[0]):
    #    y_pseudo2[:,j]=y_pseudo.iloc[:,j].values/np.sum(
    #        y_pseudo[y_pseudo.columns.difference([j])].values,axis=1)
    #y_pseudo = y_pseudo2
    
    ##distance based soft-assignment
    for j in range(centroid.shape[0]):
        ##euclidean distance
        y_pseudo[:,j]=1/np.linalg.norm(embedding-centroid[j,:],axis=1)
        ##cosine similarity
        #y_pseudo[:,j]=1/(1-cosine_similarity(embedding,centroid[j,:].reshape(1,embedding.shape[1])))[:,0]
    y_pseudo=softmax(y_pseudo,axis=1)
    
    ##auxiliary target distribution
    f = np.sum(np.square(y_pseudo)/np.sum(y_pseudo,axis=0),axis=1)
    y2 = np.square(y_pseudo)/np.sum(y_pseudo,axis=0)
    au_tar = (y2.T/f).T
    
    return au_tar, y_label,embedding

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        
##Use representation learned by CoSTA to find neighbors of genes of interest
##
def get_neighors(gene_list=None, embedding=None, target=["Vim"]):
    embedding = pd.DataFrame(embedding)
    embedding.index = gene_list
    gene_neighbors={}
    for i in target:
        distance = np.linalg.norm(embedding.values-embedding.loc[i,:].values,axis=1)
        distance = pd.DataFrame(distance)
        distance.index=gene_list
        distance = distance.sort_values(ascending=True,by=0)
        gene_neighbors[i]=distance.index.tolist()[1:51]
    return gene_neighbors


def normalize(expr = None, genes= None, h = 48, w = 48):
    n,_,a,b=expr.shape
    counts = pd.DataFrame(expr.reshape(n,a*b)).T
    counts.columns = genes

    totals = np.sum(counts,axis=1)
    bin1 = np.repeat(np.array([i for i in range(a)]), b)
    bin2 = np.tile(np.array([i for i in range(b)]), a)
    samples = pd.DataFrame({'x':bin1,'y':bin2,'total_counts':totals})

    resid_expr = NaiveDE.regress_out(samples, counts.T, 'np.log(total_counts+1)').T
    expr = resid_expr.T.values.reshape((n,1,h,w))
    return expr


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
