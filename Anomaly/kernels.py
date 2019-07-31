#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:28:46 2019

In this package, we provide all tools to build 
charatectics kernels

@author: kronert
"""
import numpy as np
### DEPRECATED
# polynomial = lambda c, d:lambda X,Y : (np.sum(X*Y,axis=1) + c)**d
#gaussian = lambda sigma: lambda X,Y : np.exp(-np.linalg.norm(X-Y,axis=1)**2/(2*sigma))
#laplacian = lambda sigma: lambda X,Y : np.exp(-sigma*np.linalg.norm(X-Y,axis=1,ord=1))

#def CCK(kernels_list, weights_list):
#    """ Compute the convex sum of two kernels
#    - Params:
#    kernels_list : list of all kernel of the sum
#    weights_list : The weights of the kernel sum
    
#    - Return :
#    A kernel, If all kernels of the sum are characteristics,
#    then the returned kernel is also characterisitic"""
#    def kernel(X,Y):
#        res = 0
#        for k,w in zip(kernels_list,weights_list):
#            res += w*k(X,Y)
#        return res
#    return  kernel


### New version

class Kernel():
    """
    This is the interface for the Kernel
    """

    def __call__(self, X,Y):
        try :
            return self.compute(X,Y)
        except :
            return self.flat_compute(X,Y)
    
    def __str__(self):
        return self.kernel_to_string()

    
    
    def kernel_to_string(self):
        return f"{self.Name} : {self.parameters}"
    
    def compute(self, X,Y):
        return None

    def flat_compute(self, X,Y):
        return None
        




class Gaussian(Kernel):

    def __init__(self, omega):
        self.Name = "Gaussian"
        self.parameters = {"omega":omega}

    def compute(self, X, Y):
        return np.exp(-np.linalg.norm(X-Y,axis=1)**2/(2*self.parameters["omega"]**2))

    def flat_compute(self, X,Y):
        return np.exp(-np.linalg.norm(X-Y)**2/(2*self.parameters["omega"]**2))





class Laplacian(Kernel):

    def __init__(self, omega):
        self.Name = "Laplacian"
        self.parameters = {"omega":omega}

    def compute(self, X, Y):
        return np.exp(-np.linalg.norm(X-Y, axis=1, ord=1)/(self.parameters["omega"]))

    def flat_compute(self, X,Y):
        return np.exp(-np.linalg.norm(X-Y, ord=1)/(self.parameters["omega"]))




class Polynomial(Kernel):

    def __init__(self, c, d):
        self.Name = "Polynomial"
        self.parameters = {"c":c, "d":d}

    def compute(self, X, Y):
        return (np.sum(X*Y,axis=1) + self.parameters["c"])**self.parameters["d"]

    def flat_compute(self, X,Y):
        return (np.sum(X*Y) + self.parameters["c"])**self.parameters["d"]


class CCK(Kernel):

    def __init__(self, ListeKernel, ListeWeights):
        self.Name = "Sum"
        self.parameters = None
        self.kernels = ListeKernel
        self.weights = ListeWeights
        self.nb = len(ListeKernel)
    
    def __call__(self, X, Y):
        sum = 0
        for k,w in zip(self.kernels,self.weights):
            sum += w*k(X,Y)
        return sum
    
    def kernel_to_string(self):
        sum = ""
        for k,w in zip(self.kernels,self.weights):
            sum += f" {w}; {k} || "
        return sum
            
