import numpy as np
import random
import math
import sys
from scipy.stats import bernoulli
from utils import random_k, positive_part, random_sparsification
from utils import loss_logistic
from utils import grad
from utils import compression_dic
from utils import compute_bit, compute_omega
import os
import pandas as pd
from scipy.stats import bernoulli


################################################################
class Standard_Newton:
    def __init__(self, oracle):
        self.oracle = oracle
    
    def step(self, x):
        '''
        x - current model weights
        return: next iterate of Standard Newton method
        '''
        lmb = self.oracle.get_reg_coef()
        d = self.oracle.get_number_of_weights()

        g = self.oracle.full_gradient(x) + lmb*x
        H = self.oracle.full_Hessian(x) + lmb*np.eye(d)
        return x - np.linalg.inv(H).dot(g) 

    def find_optimum(self, x0, n_steps=10):
        '''
        x0 - initial model weights
        n_steps - number of steps of the method 
        Implementation of Standard Newton method when we don't know the solution of the problem
        '''
        iterates = []
        iterates.append(x0)
        for k in range(n_steps):
            print(self.oracle.function_value(x0))
            x0 = self.step(x0)
            iterates.append(x0)
        self.oracle.set_optimum(iterates[-1])
        
 
    def method(self, x0, tol=10**(-14), max_iter=10):
        '''
        x0 - initial model weights
        max_iter - maximum number of steps of the method 
        Implementation of Standard Newton method
        '''
        x_opt = self.oracle.get_optimum()
        n = self.oracle.get_number_of_nodes()
        d = self.oracle.get_number_of_weights()
        func_value = []
        iterates = []
        func_value.append(self.oracle.function_value(x0))
        iterates.append(x0)
        bits = []
        global_bit = 0
        bits.append(global_bit)
        n_steps = 0
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += n*32*(d**2+d)
            bits.append(global_bit)
            
            x0 = self.step(x0)
            func_value.append(self.oracle.function_value(x0))
            iterates.append(x0)
            
        return np.array(func_value), np.array(iterates), np.array(bits)
    
#############################################################################   
class Basic_method:
    def __init__(self, oracle):
        self.oracle = oracle
        self.x_opt = oracle.get_optimum()
        self.H = oracle.full_Hessian(self.x_opt)+oracle.get_reg_coef()*np.eye(oracle.get_number_of_weights())
        
    def step(self, x):
        '''
        x - current model weights
        return: next iterate of Basic method
        '''
        lmb = self.oracle.get_reg_coef()
        d = self.oracle.get_number_of_weights()
        g = self.oracle.full_gradient(x) + lmb*x
        
        return x - np.linalg.inv(self.H).dot(g) 
    
    def method(self, x0, max_iter = 10, tol=10**(-12)):
        '''
        x0 - initial model weights
        max_iter - maximum number of steps of the method 
        Implementation of Standard Newton method
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        n = self.oracle.get_number_of_nodes()
        bits = []
        iterates = []
        func_value = []
        
        global_bit = 0
        bits.append(global_bit)  
        global_bit += d*d*n
        func_value.append(self.oracle.function_value(x0))
        iterates.append(x0)
        
        n_steps = 0
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += n*32*d
            bits.append(global_bit)
            x0 = self.step(x0)

            func_value.append(self.oracle.function_value(x0))
            iterates.append(x0)

        return np.array(func_value), np.array(iterates), np.array(bits)
    
###########################################################################    
class PositiveCase_method:
    def __init__(self, oracle):
        self.oracle = oracle
        
    def method(self, x, H, max_iter=100, k=1, eta=None, tol=10**(-14)):
        '''
        x0 - initial model weights
        max_iter - maximum number of iterations of the method
        H - initial coefficient which will approximate alpha_ij(x)
        eta - stepsize for update of H
        k - define k in random_k operator
        return: Implementation of the method for positive regularization coefficient
        '''
        x_opt = self.oracle.get_optimum()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        d = self.oracle.get_number_of_weights()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        B = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                l = i*m + j
                B += 1/N*H_old[i][j]*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))
        B = self.oracle.full_Hessian(x)
        if eta is None:
            eta = k/m 
            
        f_opt = self.oracle.function_value(x_opt)    

        func_value = []
        iterates = []
        bits = []
        global_bit = 0
        bits.append(global_bit)
        global_bit = 32*min(d,m)*n
        bits.append(global_bit)
        iterates.append(x)
        iterates.append(x)
        func_value.append(self.oracle.function_value(x))
        func_value.append(self.oracle.function_value(x))
        
        binom = np.zeros((m+1, m+1))
        for i in range(m+1):
            binom[i,0] = 1
            binom[i,i] = 1
        for i in range(2,m+1):
            for j in range(1, i):
                binom[i,j] = binom[i-1,j]+binom[i-1,j-1]
        
        bit = 32*k + np.log2(binom[m, k])
        bit = int(np.ceil(bit))

        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += ((d+1+k*d)*32+bit)*n
            bits.append(global_bit)
            global_grad = self.oracle.full_gradient(x)
            for i in range(n):
                H_old[i] = H_new[i]
                h = random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                H_new[i] = positive_part(H_old[i] + eta*h)
            x = x - np.linalg.inv(B + lmb*np.eye(d)).dot(global_grad + lmb*x)
            

            
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))


            for i in range(n):
                for j in range(m):
                    l = i*m+j
                    B += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                
        return np.array(func_value), np.array(iterates), np.array(bits)
    
########################################################################  

class PositiveCase_methodP:
    def __init__(self, oracle):
        self.oracle = oracle
        
    def method(self, x, H, p, max_iter=100, k=1, eta=None, tol=10**(-14)):
        '''
        x0 - initial model weights
        max_iter - maximum number of iterations of the method
        H - initial coefficient which will approximate alpha_ij(x)
        eta - stepsize for update of H
        k - define k in random_k operator
        return: Implementation of the method for positive regularization coefficient
        '''
        x_opt = self.oracle.get_optimum()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        d = self.oracle.get_number_of_weights()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        B = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                l = i*m + j
                B += 1/N*H_old[i][j]*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))
        B = self.oracle.full_Hessian(x)
        if eta is None:
            eta = k*p/m 
            
        f_opt = self.oracle.function_value(x_opt)    

        func_value = []
        iterates = []
        bits = []
        global_bit = 0
        bits.append(global_bit)
        global_bit = 32*min(d,m)*n
        bits.append(global_bit)
        iterates.append(x)
        iterates.append(x)
        func_value.append(self.oracle.function_value(x))
        func_value.append(self.oracle.function_value(x))
        
        binom = np.zeros((m+1, m+1))
        for i in range(m+1):
            binom[i,0] = 1
            binom[i,i] = 1
        for i in range(2,m+1):
            for j in range(1, i):
                binom[i,j] = binom[i-1,j]+binom[i-1,j-1]
        
        bit = 32*k + np.log2(binom[m, k])
        bit = int(np.ceil(bit))

        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += ((d+1+k*d)*32+bit)*n
            bits.append(global_bit)
            global_grad = self.oracle.full_gradient(x)
            for i in range(n):
                H_old[i] = H_new[i]
                if bernoulli.rvs(p):
                    h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                    global_bit += (32*k*d+bit)
                else:
                    h = np.zeros(m)
                H_new[i] = H_old[i] + h
                
            x = x - np.linalg.inv(B + lmb*np.eye(d)).dot(global_grad + lmb*x)
            
            
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))


            for i in range(n):
                for j in range(m):
                    l = i*m+j
                    B += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                
        return np.array(func_value), np.array(iterates), np.array(bits)
    
######################################################################## 
class GeneralCase_method:
    def __init__(self, oracle):
        self.oracle = oracle
        
    def method(self, x, H, gamma, max_iter=100, k=1, eta=None,tol=10**(-14)):
        '''
        x - initial model weights
        max_iter - maximum number of iterations of the method
        H - initial coefficient which will approximate alpha_ij(x)
        eta - stepsize for update of H
        k - define k in random_k operator
        return: Implementation of the method for non-negative regularization coefficient
        '''
        x_opt = self.oracle.get_optimum()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        d = self.oracle.get_number_of_weights()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        f_opt = self.oracle.function_value(x_opt)
        binom = np.zeros((m+1, m+1))
        for i in range(m+1):
            binom[i,0] = 1
            binom[i,i] = 1
        for i in range(2,m+1):
            for j in range(1, i):
                binom[i,j] = binom[i-1,j]+binom[i-1,j-1]

        B = np.zeros((d,d))
        
        if eta is None:
            eta = k/m

        A_0 = np.zeros((d,d))
        C = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                l = i*m+j
                A_0 += (2*gamma+H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                C += (2*gamma)*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
        A_k = A_0

        func_value = []
        iterates = []
        bits = []
        global_bit = 0
        bits.append(global_bit)
        global_bit = 32*min(m, 2*d)*n
        bits.append(global_bit)
        iterates.append(x)
        iterates.append(x)
        func_value.append(self.oracle.function_value(x))
        func_value.append(self.oracle.function_value(x))
        
        bit = 32*k + np.log2(binom[m, k])
        bit = int(np.ceil(bit))

        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += (32*d+32+32*k*d+bit)*n
            bits.append(global_bit)
            
            global_grad = self.oracle.full_gradient(x)
            beta = np.zeros(n)
            for i in range(n):
                beta_ik = 0
                alpha = self.oracle.alphas(x, i)
                for j in range(m):
                    c = (alpha[j]+2*gamma)/(H_old[i][j]+2*gamma)
                    if c > beta_ik:
                        beta_ik = c
                beta[i] = beta_ik
            
            beta_k = max(beta)
            B = beta_k*A_k - C
            
            x = x - np.linalg.inv(B + lmb*np.eye(d)).dot(global_grad + lmb*x)
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))
            
            for i in range(n):
                H_old[i] = H_new[i]
                h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                H_new[i] = H_old[i] + h

                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
            
            
        return np.array(func_value), np.array(iterates), np.array(bits)
    
#################################################################################################    
    
    
class GeneralCase_methodP:
    def __init__(self, oracle):
        self.oracle = oracle
        
    def method(self, x, H, p, gamma, max_iter=100, k=1, eta=None,tol=10**(-14)):
        '''
        x - initial model weights
        max_iter - maximum number of iterations of the method
        H - initial coefficient which will approximate alpha_ij(x)
        eta - stepsize for update of H
        k - define k in random_k operator
        return: Implementation of the method for non-negative regularization coefficient
        '''
        x_opt = self.oracle.get_optimum()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        d = self.oracle.get_number_of_weights()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        f_opt = self.oracle.function_value(x_opt)
        binom = np.zeros((m+1, m+1))
        for i in range(m+1):
            binom[i,0] = 1
            binom[i,i] = 1
        for i in range(2,m+1):
            for j in range(1, i):
                binom[i,j] = binom[i-1,j]+binom[i-1,j-1]

        B = np.zeros((d,d))
        
        if eta is None:
            eta = k*p/m

        A_0 = np.zeros((d,d))
        C = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                l = i*m+j
                A_0 += (2*gamma+H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                C += (2*gamma)*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
        A_k = A_0

        func_value = []
        iterates = []
        bits = []
        global_bit = 0
        bits.append(global_bit)
        global_bit = 32*min(m, 2*d)*n
        bits.append(global_bit)
        iterates.append(x)
        iterates.append(x)
        func_value.append(self.oracle.function_value(x))
        func_value.append(self.oracle.function_value(x))
        
        bit = 32*k + np.log2(binom[m, k])
        bit = int(np.ceil(bit))

        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += (32*d+32)*n
            
            
            global_grad = self.oracle.full_gradient(x)
            beta = np.zeros(n)
            for i in range(n):
                beta_ik = 0
                alpha = self.oracle.alphas(x, i)
                for j in range(m):
                    c = (alpha[j]+2*gamma)/(H_old[i][j]+2*gamma)
                    if c > beta_ik:
                        beta_ik = c
                beta[i] = beta_ik
            
            beta_k = max(beta)
            B = beta_k*A_k - C
            
            x = x - np.linalg.inv(B + lmb*np.eye(d)).dot(global_grad + lmb*x)
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))
            
            for i in range(n):
                H_old[i] = H_new[i]
                if bernoulli.rvs(p):
                    h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                    global_bit += (32*k*d+bit)
                else:
                    h = np.zeros(m)
                H_new[i] = H_old[i] + h
                
                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                    
            bits.append(global_bit)
            
            
            
        return np.array(func_value), np.array(iterates), np.array(bits)
    

############################################################################################################
class CubicMaxNewton:
    def __init__(self, oracle):
        self.oracle = oracle
        
    def subproblem_solver(self, g, H, M, tol=10**(-15), max_iter=10000):
        '''
        g - gradient of f at current point
        H - Hessian of f at current point
        M - cubic regularization coefficient
        return: shift s which minimise <g,s> + 0.5*s^T*H*s + M/6*\|s\|^3
        '''
        U, LMB, V = np.linalg.svd(H)
        t = 1
        local_func = t
        d = self.oracle.get_number_of_weights()
        for i in range(d):
            local_func -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
            
        n_steps = 0
        while np.abs(local_func) > tol and n_steps <= max_iter:
            n_steps += 1
            der_value = 1
            func_value = t
            for i in range(d):
                der_value += M*(U.dot(g))[i]**2/(LMB[i] + M/2*t)**3
                func_value -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
            t = t - func_value/der_value
            
            local_func = t
            for i in range(d):
                local_func -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
                
        t = np.sqrt(t)
        s = -np.linalg.inv(H + M*t/2*np.eye(d)).dot(g)
        return s
    
    def get_R(self):
        '''
        return: max_ij{\|a_ij\|}
        '''
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        R = 0
        for i in range(N):
            R = max(R, np.linalg.norm(self.oracle.A[i]))
            
        return R
        
    def method(self, x, H, nu, gamma, max_iter = 5000, k=1, eta=None, tol=10**(-14)):
        '''
        x - initial model weights
        max_iter - maximum number of iterations of the method
        nu - Lipschitz constant of \phi''_ij(x)
        gamma - smooth constant of \phi_ij(x)
        H - initial coefficient which will approximate alpha_ij(x)
        eta - stepsize for update of H
        k - define k in random_k operator
        return: Implementation of the method for non-negative regularization coefficient
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        
        binom = np.zeros((m+1, m+1))
        for i in range(m+1):
            binom[i,0] = 1
            binom[i,i] = 1
        for i in range(2,m+1):
            for j in range(1, i):
                binom[i,j] = binom[i-1,j]+binom[i-1,j-1]
                
        bit = 32*k + np.log2(binom[m, k])
        bit = int(np.ceil(bit))

        if eta is None:
            eta = k/m
            
        M = nu*self.get_R()**3
        
        B = np.zeros((d,d))
        A_0 = np.zeros((d,d))
        C = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                l = i*m+j
                A_0 += (2*gamma+H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                C += (2*gamma)*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
        A_k = A_0

        func_value = []
        iterates = []
        iterates.append(x)
        func_value.append(self.oracle.function_value(x))
        bits = []
        global_bit = 0
        bits.append(global_bit)

        n_steps = 0

        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += (32*d+32+32*k*d+bit)*n
            bits.append(global_bit)
            
            global_grad = self.oracle.full_gradient(x)
            beta = np.zeros(n)
            for i in range(n):
                beta_ik = 0
                alpha = self.oracle.alphas(x, i)
                for j in range(m):
                    c = (alpha[j]+2*gamma)/(H_old[i][j]+2*gamma)
                    beta_ik = max(c, beta_ik)
                beta[i] = beta_ik

            beta_k = max(beta)
            B = beta_k*A_k - C
            
            s = self.subproblem_solver(g=global_grad+lmb*x,\
                                       H=B+lmb*np.eye(d),\
                                       M=M)
            x = x + s
            
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))
            

            
            for i in range(n):
                H_old[i] = H_new[i]
                h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                H_new[i] = H_old[i] + h

                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N

        return np.array(func_value), np.array(iterates), np.array(bits)
        
        
        
        
############################################################################################################
class CubicMaxNewtonP:
    def __init__(self, oracle):
        self.oracle = oracle
        
    def subproblem_solver(self, g, H, M, tol=10**(-15), max_iter=100):
        '''
        g - gradient of f at current point
        H - Hessian of f at current point
        M - cubic regularization coefficient
        return: shift s which minimise <g,s> + 0.5*s^T*H*s + M/6*\|s\|^3
        '''
        U, LMB, V = np.linalg.svd(H)
        t = 1
        local_func = t
        d = self.oracle.get_number_of_weights()
        for i in range(d):
            local_func -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
            
        n_steps = 0
        while np.abs(local_func) > tol and n_steps <= max_iter:
            n_steps += 1
            der_value = 1
            func_value = t
            for i in range(d):
                der_value += M*(U.dot(g))[i]**2/(LMB[i] + M/2*t)**3
                func_value -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
            t = t - func_value/der_value
            
            local_func = t
            for i in range(d):
                local_func -= (U.dot(g))[i]**2/(LMB[i] + M/2*t)**2
                
        t = np.sqrt(t)
        s = -np.linalg.inv(H + M*t/2*np.eye(d)).dot(g)
        return s
    
    def get_R(self):
        '''
        return: max_ij{\|a_ij\|}
        '''
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        R = 0
        for i in range(N):
            R = max(R, np.linalg.norm(self.oracle.A[i]))
            
        return R
        
    def method(self, x, H, nu, gamma, p = 1/2, max_iter = 5000, k=1, eta=None, tol=10**(-14)):
        '''
        x - initial model weights
        max_iter - maximum number of iterations of the method
        nu - Lipschitz constant of \phi''_ij(x)
        gamma - smooth constant of \phi_ij(x)
        H - initial coefficient which will approximate alpha_ij(x)
        eta - stepsize for update of H
        k - define k in random_k operator
        return: Implementation of Cubic MaxNewton
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        
        binom = np.zeros((m+1, m+1))
        for i in range(m+1):
            binom[i,0] = 1
            binom[i,i] = 1
        for i in range(2,m+1):
            for j in range(1, i):
                binom[i,j] = binom[i-1,j]+binom[i-1,j-1]
                
        bit = 32*k + np.log2(binom[m, k])
        bit = int(np.ceil(bit))

        if eta is None:
            eta = k*p/m
            
        M = nu*self.get_R()**3
        
        B = np.zeros((d,d))
        A_0 = np.zeros((d,d))
        C = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                l = i*m+j
                A_0 += (2*gamma+H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                C += (2*gamma)*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
        A_k = A_0

        func_value = []
        iterates = []
        iterates.append(x)
        func_value.append(self.oracle.function_value(x))
        bits = []
        global_bit = 0
        bits.append(global_bit)

        n_steps = 0
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += (32*d+32)*n
            
            
            global_grad = self.oracle.full_gradient(x)
            beta = np.zeros(n)
            for i in range(n):
                beta_ik = 0
                alpha = self.oracle.alphas(x, i)
                for j in range(m):
                    c = (alpha[j]+2*gamma)/(H_old[i][j]+2*gamma)
                    if c > beta_ik:
                        beta_ik = c
                beta[i] = beta_ik

            beta_k = max(beta)
            B = beta_k*A_k - C
            
            s = self.subproblem_solver(g=global_grad+lmb*x,\
                                       H=B+lmb*np.eye(d),\
                                       M=M)
            x = x + s
            
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))
            
            for i in range(n):
                H_old[i] = H_new[i]
                if bernoulli.rvs(p):
                    h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                    global_bit += (32*k*d+bit)
                else:
                    h = np.zeros(m)
                H_new[i] = H_old[i] + h
                
                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                    
            bits.append(global_bit)
                

        return np.array(func_value), np.array(iterates), np.array(bits)
    
###############################################################################


class DINGO:
    
    def __init__(self, oracle):
        self.oracle = oracle
        
    def method(self, x, max_iter=200, tol=1e-15, phi=1e-6, theta=1e-4, rho=1e-4):
        '''
        x - initial point
        max_iter - maximum number of iterations
        tol - desired tolerance
        phi, theta, rho - parameters of DINGO
        return: implementation of DINGO
        '''
        x_opt = self.oracle.get_optimum()
        f_opt = self.oracle.function_value(x_opt)
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        x = x.copy()
        func_value = []
        bits = []
        global_bit = 0
        bits.append(global_bit)
        func_value.append(self.oracle.function_value(x))
        n_steps = 0
        cases = np.array([0,0,0])

        while func_value[-1] - f_opt > tol and max_iter >= n_steps:
            
            
            
            global_bit += 32*6*d*n
            n_steps += 1
            
            g = self.oracle.full_gradient(x) + lmb*x
            g_norm = np.linalg.norm(g)**2

            H_i = []
            H_inv = []
            H_hat = []

            h_i = np.zeros(d)
            h_inv = np.zeros(d)
            h_hat = np.zeros(d)
            full_H = self.oracle.full_Hessian(x) + lmb*np.eye(d)
            for i in range(n):
                B = self.oracle.local_Hessian(x, i)
                B += lmb*np.eye(d)

                H_i.append(B.dot(g))
                h_i += 1/n*H_i[i]
                H_inv.append(np.linalg.inv(B).dot(g))
                h_inv += 1/n*H_inv[i]
                H = np.vstack((B, phi*np.eye(d)))
                G = np.vstack((g.reshape(d,1), np.zeros((d,1))))
                H_hat.append(np.linalg.pinv(H).dot(G))
                h_hat += 1/n*H_hat[i].squeeze()


            if h_i.dot(h_inv) >= theta*g_norm:
                cases[0] += 1
                p = -h_inv
            elif h_i.dot(h_hat) >= theta*g_norm:
                cases[1] += 1
                p = -h_hat
            else:
                cases[2] += 1
                p = np.zeros(d)
                for i in range(n):
                    B = self.oracle.local_Hessian(x, i)
                    B += lmb*np.eye(d)
                    H = np.vstack((B, phi*np.eye(d)))
                    G = np.vstack((g.reshape(d,1), np.zeros((d,1))))
                    if H_hat[i].squeeze().dot(h_i) >= theta*g_norm:
                        p -= 1/n*H_hat[i].squeeze()
                    else:
                        global_bit += 32*2*d
                        p = -H_hat[i].squeeze()
                        l = -g.reshape(1,d).dot(full_H).dot(np.linalg.pinv(H)).dot(G)[0].squeeze() + theta*g_norm
                        l /= g.reshape(1,d).dot(full_H).dot(np.linalg.inv(H.T.dot(H))).dot(full_H).dot(g).squeeze()

                        p -= l*np.linalg.inv(H.T.dot(H)).dot(h_i)
                        p /= n
            global_bit += 32*2*d*n
            a = 1
            g_next = self.oracle.full_gradient(x+a*p)+lmb*(x+a*p)
            g_next_norm = np.linalg.norm(g_next)**2
            while g_next_norm > g_norm + 2*a*rho*p.dot(h_i):
                global_bit += 32*2*d*n
                a /= 2
                g_next = self.oracle.full_gradient(x+a*p)+lmb*(x+a*p)
                g_next_norm = np.linalg.norm(g_next)**2

            x = x + a*p
            func_value.append(self.oracle.function_value(x))
            bits.append(global_bit)

            
        return np.array(func_value), np.array(bits), cases
 
####################################################################

class CompressedDINGO:
    
    def __init__(self, oracle):
        self.oracle = oracle
        
        
    def comp_method(self, x):
        dim = x.shape[0]
        norm = np.linalg.norm(x, 1)
        answer = norm/dim*np.sign(x)
        return answer
        
    def method(self, x, a, max_iter=10000, tol=1e-15, phi=1e-6, theta=1e-2, rho=1e-6):
        '''
        x - initial point
        max_iter - maximum number of iterations
        a - stepsize
        tol - desired tolerance
        phi, theta, rho - parameters of DINGO
        return: implementation of DINGO
        '''
        x_opt = self.oracle.get_optimum()
        f_opt = self.oracle.function_value(x_opt)
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        x = x.copy()
        func_value = []
        bits = []
        global_bit = 0
        bits.append(global_bit)
        func_value.append(self.oracle.function_value(x))
        n_steps = 0
        cases = np.array([0,0,0])
        bit = d + 32
        print(func_value[-1])
        g = self.oracle.full_gradient(x) + lmb*x
        while func_value[-1] - f_opt > tol and max_iter >= n_steps:
            
            global_bit += 32*3*d*n
            global_bit += 3*bit*n

            n_steps += 1
            g = self.oracle.full_gradient(x) + lmb*x
            g_norm = np.linalg.norm(g)**2

            H_i = []
            H_inv = []
            H_hat = []

            h_i = np.zeros(d)
            h_inv = np.zeros(d)
            h_hat = np.zeros(d)
            full_H = self.oracle.full_Hessian(x) + lmb*np.eye(d)
            for i in range(n):
                B = self.oracle.local_Hessian(x, i)
                B += lmb*np.eye(d)

                H_i.append(self.comp_method(B.dot(g)))
                h_i += 1/n*H_i[i]
                H_inv.append(self.comp_method(np.linalg.inv(B).dot(g)))
                h_inv += 1/n*H_inv[i]
                H = np.vstack((B, phi*np.eye(d)))
                G = np.vstack((g.reshape(d,1), np.zeros((d,1))))
                H_hat.append(self.comp_method(np.linalg.pinv(H).dot(G)))
                h_hat += 1/n*H_hat[i].squeeze()


            if h_i.dot(h_inv) >= theta*g_norm:
                cases[0] += 1
                p = -h_inv
            elif h_i.dot(h_hat) >= theta*g_norm:
                cases[1] += 1
                p = -h_hat
            else:
                cases[2] += 1
                p = np.zeros(d)
                for i in range(n):
                    global_bit += 2*bit

                    B = self.oracle.local_Hessian(x, i)
                    B += lmb*np.eye(d)
                    H = np.vstack((B, phi*np.eye(d)))
                    G = np.vstack((g.reshape(d,1), np.zeros((d,1))))

                    p_i = -H_hat[i].squeeze()
                    l = theta*g_norm - h_i.squeeze().dot(H_hat[i])
                    l /= (h_i.reshape(1,d).dot(np.linalg.inv(H.T.dot(H))).dot(h_i).squeeze())
                    p_i -= l*np.linalg.inv(H.T.dot(H)).dot(h_i)
                    p += self.comp_method(p_i)/n

            x = x + a*p
            func_value.append(self.oracle.function_value(x))
            bits.append(global_bit)
        return np.array(func_value), np.array(bits), np.array(cases)
    
    
###############################################################################################
    
    
    
def dcgd(X, y, w, arg, f_opt, tol=1e-15):
    alg = 'DCGD'
    arg.eta = 1 / arg.L
  
    print('algorithm ' + alg + ' starts')
    print('eta = ', arg.eta, 'compression: ', arg.comp_method)
    print('f_opt = ', f_opt)
    dim = X.shape[1]
    num_data = y.shape[0]
    num_data_worker = int(np.floor(num_data / arg.node))
    
    loss = []
    local_grad = np.zeros((arg.node, dim))
    loss_0 = loss_logistic(X, y, w, arg)
    print('at iteration 0', 'loss =', loss_0)
    loss.append(loss_0)
    
    com_bits = [0]
    bits = 0
    
    comp_method = compression_dic[arg.comp_method]
    com_round_bit = compute_bit(dim, arg)
    
    k = 0
    while k < arg.T and loss[-1] - f_opt > tol:
        k += 1
        for i in range(arg.node):
            local_grad[i] = grad(X[i * num_data_worker:(i + 1) * num_data_worker],
                                 y[i * num_data_worker:(i + 1) * num_data_worker], w, arg)
            local_grad[i] = comp_method(local_grad[i], arg)
        gk = np.mean(local_grad, axis=0)
        assert gk.shape[0] == len(w)
        w = w - arg.eta * gk
        bits += com_round_bit
        loss_k = loss_logistic(X, y, w, arg)
        loss.append(loss_k)
        com_bits.append(bits)
        if k % 1000 == 0:
            print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits


def diana(X, y, w, arg, f_opt, tol=1e-15):
    alg = 'DIANA'
    dim = X.shape[1]
    
    omega = compute_omega(dim, arg)
    arg.alpha = 1 / (1 + omega)
    arg.eta = min(arg.alpha / (2 * arg.lamda), 2 / ((arg.L + arg.lamda) * (1 + 6 * omega / arg.node)))
    
    print('algorithm ' + alg + ' starts')
    print('eta = ', arg.eta, 'compression: ', arg.comp_method)
    print('f_opt = ', f_opt)
    
    num_data = y.shape[0]
    num_data_worker = int(np.floor(num_data / arg.node))
    
    loss = []
    local_grad = np.zeros((arg.node, dim))
    
    hs = np.zeros((arg.node, dim))
    hs_mean = np.mean(hs, axis=0)
    deltas = np.zeros((arg.node, dim))
    
    loss_0 = loss_logistic(X, y, w, arg)
    print('at iteration 0', 'loss =', loss_0)
    loss.append(loss_0)
    
    com_bits = [0]
    bits = 0
    
    comp_method = compression_dic[arg.comp_method]
    com_round_bit = compute_bit(dim, arg)
    k = 0
    while k < arg.T and loss[-1] - f_opt > tol:
        k += 1
        for i in range(arg.node):
            local_grad[i] = grad(X[i * num_data_worker:(i + 1) * num_data_worker],
                                 y[i * num_data_worker:(i + 1) * num_data_worker], w, arg)
            deltas[i] = comp_method(local_grad[i] - hs[i], arg)
            hs[i] += arg.alpha * deltas[i]
        gk = np.mean(deltas, axis=0) + hs_mean
        assert gk.shape[0] == len(w)
        hs_mean += arg.alpha * np.mean(deltas, axis=0)
        assert hs_mean.shape[0] == len(w)
        w = w - arg.eta * gk
        bits += com_round_bit
        loss_k = loss_logistic(X, y, w, arg)
        loss.append(loss_k)
        com_bits.append(bits)
        if k % 1000 == 0:
            print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits


def adiana(X, y, w, arg, f_opt, tol=1e-15):
    alg = 'ADIANA'
    dim = X.shape[1]
    
    omega = compute_omega(dim, arg)
    arg.alpha = 1 / (1 + omega)
    arg.theta_2 = 0.5
    if omega == 0:
        arg.prob = 1
        arg.eta = 0.5 / arg.L
    else:
        arg.prob = min(1, max(0.5 * arg.alpha, 0.5 * arg.alpha * (np.sqrt(arg.node / (32 * omega)) - 1)))
        arg.eta = min(0.5 / arg.L, arg.node / (64 * omega * arg.L * ((2 * arg.prob * (omega + 1) + 1) ** 2)))
    arg.theta_1 = min(1 / 4, np.sqrt(arg.eta * arg.lamda / arg.prob))
    arg.gamma = 0.5 * arg.eta / (arg.theta_1 + arg.eta * arg.lamda)
    arg.beta = 1 - arg.gamma * arg.lamda
    
    print('algorithm ' + alg + ' starts')
    print('eta = ', arg.eta, 'compression: ', arg.comp_method)
    print('f_opt = ', f_opt)
    
    dim = X.shape[1]
    num_data = y.shape[0]
    num_data_worker = int(np.floor(num_data / arg.node))
    
    zk = w
    yk = w
    wk = w
    xk = w
    
    loss = []
    local_gradx = np.zeros((arg.node, dim))
    local_gradw = np.zeros((arg.node, dim))
    hs = np.zeros((arg.node, dim))
    hs_mean = np.mean(hs, axis=0)
    deltas = np.zeros((arg.node, dim))
    deltasw = np.zeros((arg.node, dim))
    loss_0 = loss_logistic(X, y, yk, arg)
    
    print('at iteration 0', 'loss =', loss_0)
    loss.append(loss_0)
    
    com_bits = [0]
    bits = 0
    comp_method = compression_dic[arg.comp_method]
    com_round_bit = compute_bit(dim, arg)
    k=0
    while k < arg.T and loss[-1] - f_opt > tol:
        k += 1
        xk = arg.theta_1 * zk + arg.theta_2 * wk + (1 - arg.theta_1 - arg.theta_2) * yk
        for i in range(arg.node):
            local_gradx[i] = grad(X[i * num_data_worker:(i + 1) * num_data_worker],
                                  y[i * num_data_worker:(i + 1) * num_data_worker], xk, arg)
            deltas[i] = comp_method(local_gradx[i] - hs[i], arg)
            local_gradw[i] = grad(X[i * num_data_worker:(i + 1) * num_data_worker],
                                  y[i * num_data_worker:(i + 1) * num_data_worker], wk, arg)
            deltasw[i] = comp_method(local_gradw[i] - hs[i], arg)
            hs[i] += arg.alpha * deltasw[i]
        gk = np.mean(deltas, axis=0) + hs_mean
        assert gk.shape[0] == len(w)
        hs_mean += arg.alpha * np.mean(deltasw, axis=0)
        assert hs_mean.shape[0] == len(w)
        oldyk = yk
        yk = xk - arg.eta * gk
        zk = arg.beta * zk + (1 - arg.beta) * xk + (arg.gamma / arg.eta) * (yk - xk)
        change = np.random.random()
        if bernoulli.rvs(arg.prob):
            wk = oldyk
        bits += com_round_bit
        loss_k = loss_logistic(X, y, yk, arg)
        loss.append(loss_k)
        com_bits.append(bits)
        if k % 1000 == 0:
            print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits