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
import matplotlib.pyplot as plt

##1. Define Convolutional neural net of CoSTA, for Slide-seq data
##
##1. Define Convolutional neural net of CoSTA, for MERFISH data
##
class ConvNet200(torch.nn.Module):
    def __init__(self):
        super(ConvNet200, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=11,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=7,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=1,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128*6*6, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.normalize(out.view(-1, 128*6*6), p=2, dim=1)
        out = self.fc2(out)
        return out
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.normalize(out.view(-1, 128*6*6), p=2, dim=1)

        return out
    

class ConvNet85(torch.nn.Module):
    def __init__(self):
        super(ConvNet85, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=11,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=7,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128*7*7, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128*7*7), p=2, dim=1)
        out = self.fc2(out)
        return out
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128*7*7), p=2, dim=1)
        return out

    
class ConvNet85_s2(torch.nn.Module):
    def __init__(self):
        super(ConvNet85_s2, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=11,stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=7,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128*2*2, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128*2*2), p=2, dim=1)
        out = self.fc2(out)
        return out
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.normalize(out.view(-1, 128*2*2), p=2, dim=1)
        return out
    
    
class ConvNet48(torch.nn.Module):
    def __init__(self):
        super(ConvNet48, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=5,stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5,stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.Tanh(),#torch.nn.ReLU(),#
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 128)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)
        out = self.fc2(out)
        return out
    
    def forward_feature(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, 128)
        out = F.normalize(out.view(-1, 128), p=2, dim=1)
        return out