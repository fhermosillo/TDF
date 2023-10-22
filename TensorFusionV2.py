# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:52:27 2023

@author: User123
"""

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

class TensorFusionLayer(nn.Module):
    def __init__(self, I, J, R, needBias=True):
        super(TensorFusionLayer, self).__init__()
            
        
        # Parameters
        self.rank = R
        self.idims = I.copy()
        self.M = len(I)
        self.biasflag = needBias
        
        # Factors parameters
        self.W = torch.nn.ParameterList([])
        if R==-1:
            # Fully multilinear transformation
            mdims=I.copy()
            for m in range(self.M):
                mdims[m]+=1
            if isinstance(J, list):
                self.odims = J.copy()
                self.N = len(J)
                for n in range(self.N):
                    mdims.append(J[n])
            else:
                self.N = 1
                self.odims = J
                mdims.append(J)
            
            # Get einstein summation equation
            self.str_idims=""
            self.str_odims=""
            for m in range(self.M):
            	self.str_idims+=chr(97+m)
            for n in range(self.N):
            	self.str_odims+=chr(97+self.M+n)
            #ein_equ = self.str_odims + self.str_idims + "," + self.str_idims + "->" + self.str_odims
            #print("@TDFL: Einstain Notation{" + ein_equ + "}\n")
            
            # Create parameter
            self.W.append(nn.Parameter(torch.Tensor(*mdims)))
            
            # Bias term
            if needBias:
                if self.N>1:
                    self.W.append(nn.Parameter(torch.Tensor(*self.odims)))
                else:
                    self.W.append(nn.Parameter(torch.Tensor(1,self.odims)))
        else:
            self.odims = J
            for i in range(self.M):
                self.W.append(nn.Parameter(torch.Tensor(self.rank, self.idims[i] + 1, self.odims)))
            
            # Bias term
            if needBias:
                self.W.append(nn.Parameter(torch.Tensor(1,self.odims)))
        
        
        
        self.initParams()
            
            
    def initParams(self):
        if self.rank == -1:
            nn.init.xavier_normal_(self.W[0])
        else:
            for i in range(self.M):
                nn.init.xavier_normal_(self.W[i])
        
        if self.biasflag:
            self.W[-1].data.fill_(0)
    
    def forward(self, X):
        # Assertions
        torch._assert(len(X) == self.M, "Input X must contain M elements")
        
        if X[0].dim()==2:
            # Batch
            bsize = X[0].shape[0]
            ein_equ ="rij,li->rlj" # (R x Im x J), (B x Im) -> (B x R x J)
        else:
            # Sample
            ein_equ ="rij,i->rj" # (R x Im x J), Im -> (R x J)
            bsize = 1
        
        if self.rank==-1:
            # Outer product
            if X[0].is_cuda:
                Xten = torch.cat((Variable(torch.ones(bsize, 1,requires_grad=False)).type(torch.cuda.FloatTensor), X[0]), dim=1)
            else:
                Xten = torch.cat((torch.ones(bsize,1,requires_grad=False), X[0]), dim=1)
            for i in range(1,self.M):
                if X[0].is_cuda:
                    Xi = torch.cat((Variable(torch.ones(bsize, 1,requires_grad=False)).type(torch.cuda.FloatTensor), X[i]), dim=1)
                else:
                    Xi = torch.cat((torch.ones(bsize,1,requires_grad=False), X[i]), dim=1)
                if X[0].dim()==1:
                    eouter=self.str_idims[0:i]+","+self.str_idims[i]+"->"+self.str_idims[0:(i+1)]
                else:
                    eouter=chr(97+self.M+self.N)+self.str_idims[0:i]+","+chr(97+self.M+self.N)+self.str_idims[i]
                    eouter+="->"+chr(97+self.M+self.N)+self.str_idims[0:(i+1)]
                Xten=torch.einsum(eouter,Xten,Xi)
                
            
            # Multilinear transformation
            if X[0].dim()==1:
                ein_equ = self.str_idims + self.str_odims + "," + self.str_idims + "->" + self.str_odims
            else:
                ein_equ = self.str_idims + self.str_odims + "," + chr(97+self.M+self.N) + self.str_idims + "->" + chr(97+self.M+self.N) + self.str_odims
            Z = torch.einsum(ein_equ,  self.W[0], Xten)
            
            if bsize==1:
                Z = Z.squeeze() # Squeeze
            
        else:
            # Low-rank approximation ------------------------------------------
            # Multilinear transformation
            # n-mode product along the input-mode and hadamard product
            if X[0].is_cuda:
                xi = torch.cat((Variable(torch.ones(bsize, 1,requires_grad=False)).type(torch.cuda.FloatTensor), X[0]), dim=1)
            else:
                xi = torch.cat((torch.ones(bsize,1,requires_grad=False), X[0]), dim=1)
            Z = torch.einsum(ein_equ, self.W[0],xi)
            
            for m in range(1,self.M):
                # mth-factor contribution
                if X[0].is_cuda:
                    xi = torch.cat((Variable(torch.ones(bsize, 1,requires_grad=False)).type(torch.cuda.FloatTensor), X[m]), dim=1)
                else:
                    xi = torch.cat((torch.ones(bsize,1,requires_grad=False), X[m]), dim=1)
                Zm=torch.einsum(ein_equ, self.W[m], xi) # (R x J x Im), (B x Im) -> (R x B x J)
                
                # Hadamard product
                Z = torch.mul(Z, Zm)  # (R x B x J), (R x B x J) -> (R x B x J)
            
            # Summation part along the rank-mode
            Z = torch.sum(Z, dim=0)  # (R x B x J), R -> (B x J)
            Z = Z.view(-1, self.odims) # Squeeze
        
        # Bias part (Affine Transformation)
        if self.biasflag:
            Z = Z + self.W[-1]
        
        return Z
    
    def printParams(self):
        for name,param in self.named_parameters():
            print("Parameter.", name, ".shape: ", param.shape)
            print(param)