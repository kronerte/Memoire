#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:30:55 2019

@author: kronert
"""

import numpy as np
import Anomaly.utils as utils
import Anomaly.MMD as MMD


def sampleMMD_MC(methodeMMD, kernel, law_p, law_q, m, n, finalSampleSize = 1000, verbose=1):
    
    Liste_mmdl = []
    for _ in range(finalSampleSize):
        if verbose > 0:
            print(_/finalSampleSize*100, '% achevé',end='\r',flush=True)
        X = law_p(m)
        Y = law_q(n)

        MMD =  methodeMMD(kernel)
        MMD.fit(Y, X)
        Liste_mmdl.append(MMD.MMD)
        
    return np.r_[Liste_mmdl]

def moyenneMMD_Bt(methodeMMD, kernel, X, Y, n_boot, m_boot, repeat=100, verbose=1):
    mmd_list = np.zeros(repeat)
    for i in range(repeat):
        if verbose > 0:
            print(i/repeat*100, '% achevé',end='\r',flush=True)
        X_ = utils.bootstrap(X, m_boot)
        Y_ = utils.bootstrap(Y, n_boot)
        MMD = methodeMMD(kernel)
        MMD.fit(Y_, X_)
        mmd_list[i] = MMD.MMD
    return mmd_list.mean()

def meanEmbeding_Bt(kernel, X, y):
    return np.mean(kernel(y, X))


def partial_varianceOMMD_Bt(kernel, X, Y, n_boot = 100, finalSampleSize=1000, verbose = 1):
    sample = np.zeros(finalSampleSize)
    for i in range(finalSampleSize):
        if verbose > 0:
            print(i/finalSampleSize*100, '% achevé',end='\r',flush=True)
        x1 = utils.bootstrap(X, 1)
        x2 = utils.bootstrap(X, 1)
        Y_1 = utils.bootstrap(Y, n_boot)
        Y_2 =  utils.bootstrap(Y, n_boot)
        sample[i] = kernel(x1,x2) - meanEmbeding_Bt(kernel, x1, Y_1) - meanEmbeding_Bt(kernel, x2, Y_2)
    return sample.var()

def var_OMMD_Bt(kernel, X, Y, n_boot = 100, finalSampleSize=1000, verbose = 1):
    Lambda = Y.shape[0]/X.shape[0]
    if verbose > 0:
        print("Start computing first componant")
    first_componant = partial_varianceOMMD_Bt(kernel, X, Y, n_boot = n_boot, finalSampleSize=finalSampleSize, verbose = verbose)
    if verbose > 0:
        print("Start computing second componant")
    second_componant = partial_varianceOMMD_Bt(kernel, Y, X, n_boot = n_boot, finalSampleSize=finalSampleSize, verbose = verbose)
    return 2*(first_componant + 1/Lambda*second_componant)

def V1V0_Bt(kernel, X, Y, n_boot = 100, finalSampleSize=1000, verbose = 1):
    """ Renvoie le premier terme de l'élément à optimiser"""
    if verbose > 0:
        print("Start computing sigma under H0")
    sigma_H0 = np.sqrt(var_OMMD_Bt(kernel, X, X, n_boot = n_boot, finalSampleSize=finalSampleSize, verbose = verbose))
    if verbose > 0:
        print("Start computing sigma under H1")
    sigma_H1 = np.sqrt(var_OMMD_Bt(kernel, X, Y, n_boot = n_boot, finalSampleSize=finalSampleSize, verbose = verbose))
    return sigma_H0/sigma_H1

def MV_Bt(kernel, X, Y, n_boot = 100, repeat=100, n_boot_sh1=100,finalSampleSize=1000, verbose = 1):
    """ Renvoie le deuxième terme de l'élément à optimiser"""
    if verbose > 0:
        print("Start computing Mean under H1")
    mean_H1 = moyenneMMD_Bt(MMD.QMMD, kernel, X, Y, n_boot, n_boot, repeat=repeat, verbose=verbose)
    if verbose > 0:
        print("Start computing sigma under H1")
    sigma_H1 = np.sqrt(var_OMMD_Bt(kernel, X, Y, n_boot = n_boot_sh1, finalSampleSize=finalSampleSize, verbose = verbose))
    return mean_H1/sigma_H1

def score_Bt(kernel, X, Y, m, alpha,n_boot_sigma=100, n_repeat_sigma=100,n_boot_mean=100, n_repeat_mean=100,  verbose = 1):
    if verbose > 0:
        print("Start computing sigma under H0")
    sigma_H0 = np.sqrt(var_OMMD_Bt(kernel, X, X, n_boot = n_boot_sigma, finalSampleSize=n_repeat_sigma, verbose = verbose))
    if verbose > 0:
        print("Start computing sigma under H1")
    sigma_H1 = np.sqrt(var_OMMD_Bt(kernel, X, Y, n_boot = n_boot_sigma, finalSampleSize=n_repeat_sigma, verbose = verbose))
    if verbose > 0:
        print("Start computing Mean under H1")
    mean_H1 = moyenneMMD_Bt(methodeMMD, kernel, X, Y, n_boot_mean, n_boot_mean, repeat=n_repeat_mean, verbose=verbose)
    return np.sqrt(m/2)*mean_H1 - inv_phi(1-alpha)*sigma_H0/sigma_H1



def moyenneMMD_MC(methodeMMD, kernel, law_p, law_q, m, n, repeat=100, verbose=1):
    mmd_list = np.zeros(repeat)
    for i in range(repeat):
        if verbose > 0:
            print(i/repeat*100, '% achevé',end='\r',flush=True)
        X_ = law_p(m)
        Y_ = law_q(n)
        MMD = methodeMMD(kernel)
        MMD.fit(Y_, X_)
        mmd_list[i] = MMD.MMD
    return mmd_list.mean()

def meanEmbeding_MC(kernel, y,law_p, size_gen=100):
    return np.mean(kernel(y, law_p(size_gen)))


def partial_varianceOMMD_MC(kernel,  law_p, law_q, size_gen = 100, finalSampleSize=1000, verbose = 1):
    sample = np.zeros(finalSampleSize)
    for i in range(finalSampleSize):
        if verbose > 0:
            print(i/finalSampleSize*100, '% achevé',end='\r',flush=True)
        for i in range(finalSampleSize):
            x1 = law_p(1)
            x2 = law_p(1)
            sample[i] = kernel(x1,x2) -  meanEmbeding_MC(kernel, x1, law_q, size_gen = size_gen) -  meanEmbeding_MC(kernel, x2, law_q, size_gen = size_gen)
    return sample.var()


def var_OMMD_MC(kernel, law_p, law_q, size_gen = 100,Lambda = 0.1, finalSampleSize=1000, verbose = 1):
    if verbose > 0:
        print("Start computing first componant")
    first_componant = partial_varianceOMMD_MC(kernel, law_p, law_q, size_gen = size_gen, finalSampleSize=finalSampleSize, verbose = verbose)
    if verbose > 0:
        print("Start computing second componant")
    second_componant = partial_varianceOMMD_MC(kernel, law_q, law_p, size_gen = size_gen, finalSampleSize=finalSampleSize, verbose = verbose)
    return 2*(first_componant + 1/Lambda*second_componant)

def V1V0_MC(kernel, law_p, law_q, n_boot = 100, finalSampleSize=1000, verbose = 1):
    """ Renvoie le premier terme de l'élément à optimiser"""
    if verbose > 0:
        print("Start computing sigma under H0")
    sigma_H0 = np.sqrt(var_OMMD_MC(kernel, law_p, law_p, n_boot = n_boot, finalSampleSize=finalSampleSize, verbose = verbose))
    if verbose > 0:
        print("Start computing sigma under H1")
    sigma_H1 = np.sqrt(var_OMMD_MC(kernel, law_p, law_q, n_boot = n_boot, finalSampleSize=finalSampleSize, verbose = verbose))
    return sigma_H0/sigma_H1



def MV_MC(kernel, law_p, law_q, n_boot = 100, repeat=100, n_boot_sh1=100,finalSampleSize=1000, verbose = 1):
    """ Renvoie le deuxième terme de l'élément à optimiser"""
    if verbose > 0:
        print("Start computing Mean under H1")
    mean_H1 = moyenneMMD_MC(methodeMMD, kernel, law_p, law_q, n_boot, n_boot, repeat=repeat, verbose=verbose)
    if verbose > 0:
        print("Start computing sigma under H1")
    sigma_H1 = np.sqrt(var_OMMD_MC(kernel, law_p, law_q, n_boot = n_boot_sh1, finalSampleSize=finalSampleSize, verbose = verbose))
    return mean_H1/sigma_H1

def score_MC(kernel, law_p, law_q, m, alpha,n_boot_sigma=100, n_repeat_sigma=100,n_boot_mean=100, n_repeat_mean=100,  verbose = 1):
    if verbose > 0:
        print("Start computing sigma under H0")
    sigma_H0 = np.sqrt(var_OMMD_MC(kernel, law_p, law_p, size_gen = n_boot_sigma, finalSampleSize=n_repeat_sigma, verbose = verbose))
    if verbose > 0:
        print("Start computing sigma under H1")
    sigma_H1 = np.sqrt(var_OMMD_MC(kernel, law_p, law_q, size_gen = n_boot_sigma, finalSampleSize=n_repeat_sigma, verbose = verbose))
    if verbose > 0:
        print("Start computing Mean under H1")
    mean_H1 = moyenneMMD_MC(MMD.OMMD, kernel, law_p, law_q, n_boot_mean, n_boot_mean, repeat=n_repeat_mean, verbose=verbose)
    return np.sqrt(m/2)*mean_H1 -  utils.inv_phi(1-alpha)*sigma_H0/sigma_H1



def threshold_MC(kernel, law_p, law_q, m, n, alpha, size_gen = 100, finalSampleSize=1000, verbose = 1):
    sigma = np.sqrt( var_OMMD_MC(kernel, law_p, law_q, size_gen = size_gen,Lambda = n/m, finalSampleSize=finalSampleSize, verbose = verbose))
    invPhiAlpha = utils.inv_phi(1-alpha)
    return sigma*invPhiAlpha/np.sqrt(m)




def puissance_MC(methodeMMD, kernel, law_p, law_q, m, n, threshold, repeat=1000):
    mmd =sampleMMD_MC(methodeMMD, kernel, law_p, law_q, m, n, finalSampleSize = 1000, verbose=1)
    return (mmd<threshold).mean()


    
    
