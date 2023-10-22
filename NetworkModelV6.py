# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:48:24 2023

@author: User123
"""

from torch import nn
import torch
import TensorFusionV2 as tf
import tensorly.decomposition as tl

class NetworkModel(nn.Module):
    def __init__(self,I,J,R,trainParams):
        super(NetworkModel,self).__init__()
        
        torch._assert(len(I) > 1, "There must be at least two modals")
        
        # TDF Layer
        self.tfl = tf.TensorFusionLayer(I,J,R)
        self.bnF = torch.nn.BatchNorm1d(J)
        
        # Hidden Layers
        self.fc1 = torch.nn.Linear(J, int(2*J))
        self.bn1 = torch.nn.BatchNorm1d(int(2*J))
        #self.fc2 = torch.nn.Linear(int(2*J),int(J/2))
        #self.bn2 = torch.nn.BatchNorm1d(int(J/2))
        self.dropout = nn.Dropout(p=0.5)
        
        # Activation layers
        self.relu = torch.nn.ReLU()
        
        # Output layers
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        
        # Output layer (Classification)
        self.class_layer = nn.Sequential(
            #nn.Linear(int(J/2), 6),  # 6 clases: Anomaly, Pickup truck, motorcycle, sedan, truck, trailer
            nn.Linear(int(J*2), int(J*4)),
            nn.BatchNorm1d(int(J*4)),
            nn.Linear(int(J*4), int(J)),
            nn.BatchNorm1d(int(J)),
            nn.Linear(int(J), int(6)),
            nn.Dropout(p=0.3),
            nn.Softmax(dim=1)
        )

        # Output layer (Regression)
        self.occ_layer = nn.Sequential(
            #nn.Linear(int(J/2), 2), # 2 classes: Unoccluded, Occluded
            nn.Linear(int(J*2), int(J/2)),
            nn.Linear(int(J/2), 2),
            nn.Softmax(dim=1)
        )
        
        self.TrainParams = trainParams
        
        
    def forward(self, X):
        torch._assert(len(X)==self.tfl.M, "len(X) must be equals to M")
        
        #tl.tucker(tensor, rank, fixed_factors=None, n_iter_max=100, init='svd', return_errors=False, svd='truncated_svd', tol=0.0001, random_state=None, mask=None, verbose=False)
        
        z = self.relu(self.tfl(X))
        z = self.bnF(z)
        z = self.relu(self.fc1(z))
        z = self.bn1(z)
        #z = self.relu(self.fc2(z))
        #z = self.bn2(z)
        #z = self.dropout(z)
        
        y_clas_pred= self.class_layer(z)
        y_occ_pred = self.occ_layer(z)
        
        return y_clas_pred, y_occ_pred
        
        