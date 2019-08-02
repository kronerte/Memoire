#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:28:46 2019

In this package, we provide all tools to build 
charatectics kernels

@author: kronert
"""
import numpy as np
import copy
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

    def __add__(self, other):
        return CCK([self, other], [1,1])

    def __radd__(self, other):
        return CCK([self, other], [1,1])


    def __rmul__(self, other):
        if type(other) in (int, float):
            return CCK([self], [other])

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Kernel):
            return (self.Name == other.Name and self.parameters == other.parameters)

        return False






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
        self.reduce_flatten()
    
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


    def flatten(self):
        new_kernels_list = []
        new_weights_list = []

        kernels_to_visit = copy.deepcopy(self.kernels)
        corresponding_weights = copy.deepcopy(self.weights)

        nodes_to_visit = list(zip(kernels_to_visit, corresponding_weights))

        for k,w in  nodes_to_visit:
            if k.Name == "Sum":
                for (k_child, w_child) in zip(k.kernels, k.weights):
                    nodes_to_visit.append((k_child, w*w_child))
            else:
                new_kernels_list.append(k)
                new_weights_list.append(w)
        
        self.kernels = copy.deepcopy(new_kernels_list)
        self.weights = copy.deepcopy(new_weights_list)

    def reduce(self):
        new_kernels_list = []
        new_weights_list = []
        
        kernels_to_visit = copy.deepcopy(self.kernels)
        corresponding_weights = copy.deepcopy(self.weights)

        nodes_to_visit = list(zip(kernels_to_visit, corresponding_weights))
        
        for k,w in  nodes_to_visit:
            if k not in new_kernels_list:
                new_kernels_list.append(k)
                new_weights_list.append(w)
            else :
                ik = new_kernels_list.index(k)
                new_weights_list[ik] += w

        self.kernels = copy.deepcopy(new_kernels_list)
        self.weights = copy.deepcopy(new_weights_list)

    def reduce_flatten(self):
        self.flatten()
        self.reduce()

        


    
    

            
            
