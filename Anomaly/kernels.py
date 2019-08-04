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
from scipy import stats
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



    ### THEORICAL COMPUTATION
    def normal_score(self, mu_p, sigma_p, mu_q, sigma_q, m, n):
        return (np.sqrt(self.normal_variance(mu_p, sigma_p, mu_p, sigma_p, m, n)
        /self.normal_variance(mu_p, sigma_p, mu_q, sigma_q, m, n))*stats.norm.ppf(1-alpha)
        - np.sqrt(m/2*self.normal_MMD(mu_p, sigma_p, mu_q, sigma_q)/self.normal_variance(mu_p, sigma_p, mu_q, sigma_q, m, n)))

    def normal_power(self, mu_p, sigma_p, mu_q, sigma_q, m, n):
        return stats.norm.sf((np.sqrt(m)*(self.normal_thresohld(mu_p, sigma_p, alpha, m, n)
                           - self.normal_MMD(mu_p, sigma_p, mu_q, sigma_q)))
                          /np.sqrt(self.normal_variance(mu_p, sigma_p, mu_q, sigma_q, m, n)))

    
    def normal_thresohld(self, mu_p, sigma_p, alpha, m, n):
        return (np.sqrt(2*self.normal_variance(mu_p, sigma_p, mu_q, sigma_q, m, n))
                *stats.norm.ppf(1-alpha)/np.sqrt(m))

    

    def normal_variance(self, mu_p, sigma_p, mu_q, sigma_q, m, n):
        return 2*(normal_partial_variance(self, mu_p, sigma_p, mu_q, sigma_q) 
                + m/n*normal_partial_variance(self, mu_q, sigma_q, mu_p, sigma_p))

    def normal_MMD(self, mu_p, sigma_p, mu_q, sigma_q ):
        return None

    def normal_partial_variance(self, mu_p, sigma_p, mu_q, sigma_q ):
        return None







class Gaussian(Kernel):

    def __init__(self, omega):
        self.Name = "Gaussian"
        self.parameters = {"omega":omega}

    def compute(self, X, Y):
        return np.exp(-np.linalg.norm(X-Y,axis=1)**2/(2*self.parameters["omega"]**2))

    def flat_compute(self, X,Y):
        return np.exp(-np.linalg.norm(X-Y)**2/(2*self.parameters["omega"]**2))


    ### THEORICAL COMPUTATION
    def normal_MMD(self, mu_p, sigma_p, mu_q, sigma_q ):
        omega = self.parameters['omega']
        (np.sqrt(omega**2/(2*sigma_p**2+omega**2))
                   + np.sqrt(omega**2/(2*sigma_q**2+omega**2))
                   -2*np.sqrt(omega**2/(sigma_p**2+sigma_q**2+omega**2))*np.exp(-(mu_p-mu_q)**2/(2*(sigma_p**2+sigma_q**2+omega**2))))

    def normal_partial_variance(self, mu_p, sigma_p, mu_q, sigma_q ):
        return (self.normal_second_order_cross_normilized_kernel(mu_p, sigma_p, mu_q, sigma_q) 
                - self.normal_first_order_cross_normilized_kernel(mu_p, sigma_p, mu_q, sigma_q))

    def normal_second_order_cross_normilized_kernel(self, mu_p, sigma_p, mu_q, sigma_q):
        omega = self.parameters['omega']
        return (
            np.sqrt(omega**2/(4*sigma_p**2 + omega**2))
                
                + 2*omega**2/(np.sqrt(omega**2+sigma_q**2)*np.sqrt(omega**2+sigma_q**2+2*sigma_p**2))*np.exp(-(mu_p-mu_q)**2/(omega**2+sigma_q**2+2*sigma_p**2))
            
                + 2*omega**2/(omega**2+sigma_q**2+sigma_p**2)*np.exp(-(mu_p-mu_q)**2/(omega**2+sigma_q**2+sigma_p**2))
            
                -4*np.sqrt(omega**4/(sigma_p**4+3*sigma_p**2*omega**2+2*sigma_p**2*sigma_q**2+omega**4+sigma_q**2*omega**2))
                *np.exp(-(mu_p-mu_q)**2/(2*((sigma_p**4+3*sigma_p**2*omega**2+2*sigma_p**2*sigma_q**2+omega**4+sigma_q**2*omega**2)/(2*sigma_p**2+omega**2))))
            )

    def normal_first_order_cross_normilized_kernel(self, mu_p, sigma_p, mu_q, sigma_q):
        omega = self.parameters['omega']
         return (np.sqrt(omega**2/(2*sigma_p**2+omega**2)) 
            - 2*np.sqrt(omega**2/(sigma_p**2+sigma_q**2+omega**2))*np.exp(-(mu_p-mu_q)**2/(2*(sigma_p**2+sigma_q**2+omega**2))))


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

    
    ### THEORICAL COMPUTATION
    def normal_MMD(self, mu_p, sigma_p, mu_q, sigma_q ):
        beta = np.vect(self.weights).reshape((-1,1))
        return beta.T.dot(self.normal_vect_MMD(mu_p, sigma_p, mu_q, sigma_q))
    
    def normal_vect_MMD(self, mu_p, sigma_p, mu_q, sigma_q ): 
        res = np.zeros(len(self.nb))
        for i in range(nb):
            res[i] = self.kernels[i].normal_MMD(mu_p, sigma_p, mu_q, sigma_q)


    def normal_partial_variance(self, mu_p, sigma_p, mu_q, sigma_q ):
        beta = np.vect(self.weights).reshape((-1,1))
        return beta.T.dot(self.normal_partial_variance_Matrix(mu_p, sigma_p, mu_q, sigma_q )).dot(beta)


    def normal_partial_variance_Matrix(self, mu_p, sigma_p, mu_q, sigma_q):
        Mat = np.zeros((self.nb, self.nb)):

        for i in range(self.nb):
            for j in range(i):
                res = self.normal_covariance(self.kernels[i], self.kernels[j], mu_p, sigma_p, mu_q, sigma_q)
                Mat[i,j] = res
                Mat[j, i] = res 
            res = kernels[i].normal_partial_variance(mu_p, sigma_p, mu_q, sigma_q )
            Mat[i, i] = res 
        return Mat

    def normal_covariance(self, kernel1, kernel2, mu_p, sigma_p, mu_q, sigma_q):
        return (self.normal_cross_cross_normilized_kernel(kernel1, kernel2, mu_p, sigma_p, mu_q, sigma_q)
                - kernel1.normal_first_order_cross_normilized_kernel(mu_p, sigma_p, mu_q, sigma_q)*
                kernel2.normal_first_order_cross_normilized_kernel(mu_p, sigma_p, mu_q, sigma_q))

    def normal_cross_cross_normilized_kernel(self, kernel1, kernel2, mu_p, sigma_p, mu_q, sigma_q):
        if (kernel1.Name=="Gaussian" and kernel2.Name=="Gaussian"):
            omega1 = kernel1.parameters['omega']
            omega2 = kernel2.parameters['omega']
            omega = np.sqrt((omega1**2*omega2**2)/(omega1**2 + omega2**2))
            omegaSigma = np.sqrt(((omega1**2+sigma2**2)*(omega2**2+sigma2**2))/((omega1**2+sigma2**2) + (omega2**2+sigma2**2)))
            
            return (
            np.sqrt(omega**2/(2*sigma_p**2 + omega**2))
                
                + 2*(np.sqrt(omega1**2/(omega1**2+sigma_q**2))*np.sqrt(omega2**2/(omega2**2+sigma_q**2))
                *np.sqrt(omegaSigma**2/(omegaSigma**2+sigma_p**2)))*np.exp(-(mu_p-muq)**2/(2*(omegaSigma**2+sigma_p**2)))

                + 2*np.sqrt(omega1**2/(omega1**2+sigma_q**2+sigma_p**2))*np.exp(-(mu_p-muq)**2/(2*(omega1**2+sigma_q**2+sigma_p**2)))
                *np.sqrt(omega2**2/(omega2**2+sigma_q**2+sigma_p**2))*np.exp(-(mu_p-muq)**2/(2*(omega2**2+sigma_q**2+sigma_p**2)))

                -2*np.sqrt(omega1**2*omega2**2/(sigma_p**4+sigma1**2*omega1**2+2*sigma_p**2*omega2**2+2*sigma_p**2*sigma_q**2+omega1**2*omega2**2+sigma_q**2*omega1**2))
                *np.exp(-(mu_p-muq)**2/(2*((sigma_p**4+sigma_p**2*omega1**2+2*sigma_p**2*omega2**2+2*sigma_p**2*sigma_q**2+omega1**2*omega2**2+sigma_q**2*omega1**2)/(2*sigma_p**2+omega1**2))))
            
                -2*np.sqrt(omega2**2*omega1**2/(sigma_p**4+sigma_p**2*omega2**2+2*sigma_p**2*omega1**2+2*sigma_p**2*sigma_q**2+omega2**2*omega1**2+sigma_q**2*omega2**2))
                *np.exp(-(mu_p-muq)**2/(2*((sigma_p**4+sigma_p**2*omega2**2+2*sigma_p**2*omega1**2+2*sigma_p**2*sigma_q**2+omega2**2*omega1**2+sigma_q**2*omega2**2)/(2*sigma_p**2+omega2**2))))
            )



        


    
    

            
            
