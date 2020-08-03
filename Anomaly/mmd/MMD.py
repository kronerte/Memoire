#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:31:14 2019

@author: kronert
"""
import numpy as np
# MMD Estimators


class QMMD():
    def __init__(self, kernel):
        self.kernel = kernel
    
    def fit(self, Y, X):
        m = X.shape[0]
        res = 0
        
        X_1,X_2 = np.meshgrid(X,X)
        m_,n_ = np.meshgrid(np.arange(0,m),np.meshgrid(np.arange(0,m)))
        X_, Y_ = X_1[n_!=m_].reshape(-1,1) ,X_2[n_!=m_].reshape(-1,1)
        res += self.kernel(X_,Y_).sum()/(m*(m-1))
        
        X_1,X_2 = np.meshgrid(Y,Y)
        m_,n_ = np.meshgrid(np.arange(0,m),np.meshgrid(np.arange(0,m)))
        X_, Y_ = X_1[n_!=m_].reshape(-1,1) ,X_2[n_!=m_].reshape(-1,1)
        res += self.kernel(X_,Y_).sum()/(m*(m-1))
        
        X_1,X_2 = np.meshgrid(X,Y)
        m_,n_ = np.meshgrid(np.arange(0,m),np.meshgrid(np.arange(0,m)))
        X_, Y_ = X_1[n_!=m_].reshape(-1,1) ,X_2[n_!=m_].reshape(-1,1)
        res -= 2*self.kernel(X_,Y_).sum()/(m*(m-1))
        
        self.MMD = res

class LMMD():
    def __init__(self, kernel):
        self.kernel = kernel
    
    def fit(self, Y, X):
        m = X.shape[0]
        res = 0
        
        X_, Y_ = X[0::2,:],X[1::2,:]
        res += self.kernel(X_,Y_).sum()
        
        
        X_, Y_ = Y[0::2,:],Y[1::2,:]
        res += self.kernel(X_,Y_).sum()
        
        X_, Y_ = X[0::2,:],Y[1::2,:]
        res -= self.kernel(X_,Y_).sum()
        
        X_, Y_ = Y[0::2,:],X[1::2,:]
        res -= self.kernel(X_,Y_).sum()
        self.MMD = 2*res/m


class OMMD():
    def __init__(self, kernel, X=None):
        self.kernel = kernel
        self.X = []
        self.Y = []
        self.m = 0
        self.n = 0
        self.estim_normP = 0
        self.sumNormQ = 0
        self.sumPS = 0
        self.MMD = 0
        if X is not None:
            self.X = X
            self.m = X.shape[0]
            X_, Y_ = X[0::2,:],X[1::2,:]
            self.estim_normP = 2/self.m*self.kernel(X_,Y_).sum()
            self.MMD = self.estim_normP 
           
    def update(self,y1,y2):
        self.sumNormQ += self.kernel(y1,y2)
        
        self.sumPS += self.kernel(self.X,y1).sum() + self.kernel(self.X,y2).sum()
        
        self.n += 2
        
        self.MMD = (self.estim_normP + 2/self.n * self.sumNormQ - 2/(self.n*self.m)*self.sumPS)[0]
        
    def fit(self, Y, X=None):
        if X is not None:
            self.X = X
            X_, Y_ = X[0::2,:],X[1::2,:]
            self.m = X.shape[0]
            self.estim_normP = 2/self.m*self.kernel(X_,Y_).sum()
        for i in range(0,Y.shape[0]//2):
            self.update(Y[2*i].reshape((1,-1)),Y[2*i+1].reshape((1,-1)))