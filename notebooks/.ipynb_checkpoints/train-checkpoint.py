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
from utils import *

def train_costa(net = None, expr_array = None, num_epoch = 6, batch_size = 170, t1 = 0.8, t2 = 1.2, cluster_method = 'GMM'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_all_tensor = torch.tensor(expr_array).float()

    y_pred = net.forward_feature(X_all_tensor)
    y_pred = torch.Tensor.cpu(y_pred).detach().numpy()
    au_tar, y_label, embedding = evaluation(y_pred,n_neighbors=5,min_dist=0.0,
                                            num_cluster=10,cluster_method=cluster_method) 

    #opt = torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9)
    opt = torch.optim.Adam(net.parameters())
    
    ##visualization without training
    original = y_label.copy()
    embedding = umap.UMAP(n_neighbors=5, min_dist=0, n_components=2,
                          metric='correlation').fit_transform(y_pred)

    embedding = pd.DataFrame(embedding)
    embedding.columns=['UMAP1','UMAP2']
    embedding["Proton"]=original
    f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue="Proton",
                 fit_reg=False,legend=False,scatter_kws={'s':15})
    
    for i in list(set(y_label)):
        plt.annotate(i, 
                     embedding.loc[embedding['Proton']==i,['UMAP1','UMAP2']].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=15, weight='bold')
    plt.title("Initial Clusters")
    # f.savefig("initial_umap.jpeg",dpi=450)

    ##Training
    NMI_history = []
    for k in range(1,num_epoch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        old_label=y_label.copy()
        net.to(device)

        X_train, X_test, y_train, y_test = train_test_split(expr_array, au_tar, test_size=0.001)
        X_tensor=torch.tensor(X_train).float()
        y_tensor = torch.tensor(y_train).float()
        n = y_train.shape[0]
        for j in range(n//batch_size):
            inputs = X_tensor[j*batch_size:(j+1)*batch_size,:,:,:].to(device)
            outputs = y_tensor[j*batch_size:(j+1)*batch_size,:].to(device)
            opt.zero_grad()
            output = net.forward(inputs)
            #loss = Loss(output, outputs)
            loss = bi_tempered_logistic_loss(output, outputs,t1, t2)
            loss.backward()
            opt.step()

        #if k%5==0:
        net.to(torch.device("cpu"))
        y_pred = net.forward_feature(X_all_tensor)
        y_pred = torch.Tensor.cpu(y_pred).detach().numpy()
        au_tar, y_label, embedding = evaluation(y_pred,n_neighbors=5,min_dist=0.0,
                                                num_cluster=10,cluster_method=cluster_method) 
        cm = confusion_matrix(old_label, y_label)
        au_tar=au_tar[:,np.argmax(cm,axis=1).tolist()]
        nmi = round(normalized_mutual_info_score(old_label, y_label),5)
        print("NMI"+"("+str(k)+"|"+str(k-1)+"): "+str(nmi))
        NMI_history.append(nmi)

    ##save mode
    # torch.save(net, "../../results/merfish_models")
    # net = torch.load("../../results/merfish_models")
    
    embedding = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=2,
                      metric='correlation').fit_transform(y_pred)

    embedding = pd.DataFrame(embedding)
    embedding.columns=['UMAP1','UMAP2']
    embedding["Proton"]=original
    f=sns.lmplot(x='UMAP1', y='UMAP2',data=embedding,hue="Proton",
                 fit_reg=False,legend=False,scatter_kws={'s':15})
    for i in list(set(y_label)):
        plt.annotate(i, 
                     embedding.loc[embedding['Proton']==i,['UMAP1','UMAP2']].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     size=15, weight='bold')
    plt.title("Clusters after training")

    #f.savefig("trained_umap.jpeg",dpi=450)
    
    return y_pred, au_tar, embedding, NMI_history
