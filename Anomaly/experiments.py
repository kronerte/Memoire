#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:31:26 2019

@author: kronert
"""
import numpy as np
import pandas as pd
import inspect
import pickle
import Anomaly.estimators as estimators
import Anomaly.MMD as MMD
from scipy.stats import shapiro, norm
import matplotlib.pyplot as plt


class Experiment():
    def __init__(self, **kargs):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
    
        for arg, val in values.items():
            setattr(self, arg, val)
    def save(self, fileName):
        with open(fileName, "wb") as f:
            pickle.dump(self.__dict__,f)
    def load(self, fileName):
        with open(fileName, "rb") as f:
            self.__dict__ = pickle.load(f)
    def process(self):
        print("Not implemented yet")
    def view(self):
        print("Not implemented yet")
        
        
class ConvergenceInDistribution(Experiment):
    def __init__(self,law_H0, law_H1, Lambda, kernel, kernel_params, m, n):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
    
        for arg, val in values.items():
            setattr(self, arg, val)
    def process(self):
        # Bootstrap estimation
        X = self.law_H0(self.m)
        Y = self.law_H1(self.n)
        self.menEst = estimators.moyenneMMD_Bt(methodeMMD = MMD.OMMD, kernel = self.kernel, X=X, Y=Y, n_boot = 100, m_boot=100, repeat=100, verbose=1)
        self.var = estimators.var_OMMD_Bt(self.kernel, X, Y, n_boot = 100, finalSampleSize=1000, verbose = 1)/self.m
        
        self.sample = estimators.sampleMMD_MC(methodeMMD= MMD.OMMD, kernel = self.kernel, law_p = self.law_H0, law_q = self.law_H1, m = self.m, n = self.n, finalSampleSize = 1000, verbose=1)
        self.shapiro = shapiro(self.sample)
    def view(self):
        print("estimated mean = ", self.menEst)
        print("sampled mean = ", self.sample.mean())
        print("estimated var = ", self.var)
        print("sampled var = ", self.sample.var())
        print("**************************************")
        print("Shapiro Test results (p-value)", self.shapiro)
        sample_renormed = (self.sample-self.sample.mean())/self.sample.std()
        plt.hist(sample_renormed,density=True,bins=15)
        x = np.linspace(-4,4,1000)
        plt.plot(x,norm.pdf(x))
        
class Shapiro(Experiment):
    def __init__(self,law_H0, law_H1, kernel, ListMN):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
    
        for arg, val in values.items():
            setattr(self, arg, val)
    def process(self):
        self.shapiros = []
        for m,n in self.ListMN:
            self.shapiros.append(estimators.sampleMMD_MC(methodeMMD= MMD.OMMD, 
                                                         kernel = self.kernel, law_p = self.law_H0, law_q = self.law_H1, m = m,
                                                         n = n, finalSampleSize = 1000, verbose=1)[1])
    def view(self):
        print(self.ListMN)
        plt.plot(self.shapiros)
        plt.title("p-value for a Shapiro test")



"""def line_optimisation(law_H0, law_H1, Lambda, kernel, kernel_params, m, alpha):
    size = len(kernel_params)
    sigma_H0 = np.zeros(size)
    sigma_H1 = np.zeros(size)
    moyenne_H1 = np.zeros(size)
    V1V0_ = np.zeros(size)
    MV_ = np.zeros(size)
    opt = np.zeros(size)
    for i in range(size):
        param = kernel_params[i]
        X = law_H0(10000)
        Y = law_H1(10000)
                    
        sigma_H0[i] = np.sqrt(var_H(kernel(param), X, X, Lambda, size=100))
        sigma_H1[i] = np.sqrt(var_H(kernel(param), X, Y, Lambda, size=100))
        moyenne_H1[i] = moyenne(kernel(param), X, Y, repeat=100)
        print(f"Progression {((i+1)/size)*100:.0f}%", end="\r", flush=True)
        
    V1V0_ = V1V0(sigma_H0, alpha, sigma_H1)
    MV_ = MV(m, moyenne_H1, sigma_H1)
    
    opt = V1V0_ - MV_
    
    return sigma_H0, sigma_H1, moyenne_H1, V1V0_, MV_, opt

# Statistical Test
def normality_test(kernel, law_p, law_q, m, n):
    sample = sampleMMD(kernel,law_p, law_q, m, n ,repeat=1000)
    
    X = law_p(100000)
    Y = law_q(100000)
    var = var_H(kernel, X, Y, alpha=n/m, size=1000)
    mean = moyenne(kernel,X, Y)
    sample_renormed = (sample-mean)*np.sqrt(m/(2*var))
    plt.hist(sample_renormed,density=True,bins=15);
    x = np.linspace(-4,4,1000)
    plt.plot(x,norm.pdf(x))
    return shapiro(sample)

# Experiments
def one_line_experiment(law_p, law_q, kernel, kernel_params, m=10000,Lambda=.1, alpha=0.05):
    
    sigma_H0, sigma_H1, moyenne_H1, V1V0_, MV_, opt = line_optimisation(law_p, 
                                                                        law_q,Lambda, kernel, kernel_params, m, alpha)
    plt.plot(kernel_params, sigma_H0, label="sigma_H0")
    plt.plot(kernel_params, sigma_H1, label="sigma_H1")
    plt.plot(kernel_params, moyenne_H1, label="moyenne_H1")
    plt.plot(kernel_params, V1V0_, label="V1V0")
    plt.plot(kernel_params, MV_, label="MV")
    plt.plot(kernel_params, opt, label="opt")
    plt.legend()
    
    
    
    
class convergence_in_distribution():
    def __init__(self, law_p, law_q, m, n, kernel, size_sample, seed=42):
        self.law_p = law_p
        self.law_q = law_q
        self.n = n
        self.m = m
        self.kernel = kernel
        self.size_sample = size_sample
        
    def process(self):
        self.sample = sampleMMD_MC(self.kernel,self.law_p, self.law_q, self.m, self.n , self.size_sample=1000, seed = 42)
        self.predictedMean
        self.predictedVariance
        self.shapiro = shapiro(self.sample)
        
        
    def save(self):
        
    def show(self):
        sample_renormed = (self.sample-self.sample.mean())/self.sample.std()
        plt.hist(sample_renormed,density=True,bins=15)
        x = np.linspace(-4,4,1000)
        plt.plot(x,norm.pdf(x))
        print("estimated mean = ", self.predictedMean)"""