#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:28:46 2019

In this package, we provide all tools to build 
charatectics kernels

@author: kronert
"""
import numpy as np

polynomial = lambda c, d:lambda x,y : (np.sum(x*y,axis=1) + c)**d
gaussian = lambda sigma: lambda x,y : np.exp(-np.linalg.norm(x-y,axis=1)**2/(2*sigma))
laplacian = lambda sigma: lambda x,y : np.exp(-sigma*np.linalg.norm(x-y,axis=1,ord=1))

def CCK(kernels_list, weights_list):
    """ Compute the convex sum of two kernels
    - Params:
    kernels_list : list of all kernel of the sum
    weights_list : The weights of the kernel sum
    
    - Return :
    A kernel, If all kernels of the sum are characteristics,
    then the returned kernel is also characterisitic"""
    def kernel(x,y):
        res = 0
        for k,w in zip(kernels_list,weights_list):
            res += w*k(x,y)
        return res
    return  kernel
