import numpy as np
from scipy.stats import bernoulli
from utils import random_k, positive_part, random_sparsification
from utils import loss_logistic, grad
from utils import compression_dic
from utils import compute_bit, compute_omega


################################################################
class Standard_Newton:
    def __init__(self, oracle):
        '''
        -------------------------------------------------
        This class is created to simulate Newton's method
        -------------------------------------------------
        '''
        self.oracle = oracle
    
    def step(self, x):
        '''
        -----------------------------------
        perform one step of Newton's method
        -----------------------------------
        input:
        x - current model weights
        
        return: 
        numpy array - next iterate of Newton's method
        '''
        lmb = self.oracle.get_reg_coef()
        d = self.oracle.get_number_of_weights()

        g = self.oracle.full_gradient(x) + lmb*x
        H = self.oracle.full_Hessian(x) + lmb*np.eye(d)
        s = np.linalg.solve(H, g)
        return x - s 

    def find_optimum(self, x0, n_steps=10, verbose=True):
        '''
        -------------------------------------------------------------------------------------
        Implementation of Standard Newton method in order to find the solution of the problem
        -------------------------------------------------------------------------------------
        input:
        x0 - initial model weights
        n_steps - number of steps of the method 
        verbose - if True, then function values in each iteration are printed
        
        return:
        set the optimum to the problem
        '''
        iterates = []
        iterates.append(x0)
        for k in range(n_steps):
            if verbose:
                print(self.oracle.function_value(x0))
            x0 = self.step(x0)
            iterates.append(x0)
        self.oracle.set_optimum(iterates[-1])
        
 
    def method(self, x0, tol=10**(-14), max_iter=10, verbose=True):
        '''
        ----------------------------------------
        Implementation of Standard Newton method
        ----------------------------------------
        input:
        x0 - initial model weights
        tol - desired tolerance of the solution
        max_iter - maximum number of iterations of the method 
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        x_opt = self.oracle.get_optimum()
        n = self.oracle.get_number_of_nodes()
        d = self.oracle.get_number_of_weights()
        func_value = []
        iterates = []
        func_value.append(self.oracle.function_value(x0))
        iterates.append(np.linalg.norm(x0-x_opt))
        bits = []
        global_bit = 1
        bits.append(global_bit)
        n_steps = 0
        
        if verbose:
            print(func_value[-1])
            
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += 32*(d**2+d)
            bits.append(global_bit)
            
            x0 = self.step(x0)
            func_value.append(self.oracle.function_value(x0))
            iterates.append(np.linalg.norm(x0-x_opt))
            
            if verbose:
                print(func_value[-1])
            
        return np.array(func_value), np.array(iterates), np.array(bits)
    
#############################################################################   
class Newton_Star:
    def __init__(self, oracle):
        '''
        ---------------------------------------------------------
        This class is created to simulate NEWTON-STAR (NS) method 
        ---------------------------------------------------------
        '''
        self.oracle = oracle
        self.x_opt = oracle.get_optimum()
        self.H = oracle.full_Hessian(self.x_opt)+oracle.get_reg_coef()*np.eye(oracle.get_number_of_weights())
        
    def step(self, x):
        '''
        ----------------------
        perform one step of NS
        ----------------------
        input:
        x - current model weights
        
        return: 
        numpy array - next iterate of NS
        '''
        lmb = self.oracle.get_reg_coef()
        d = self.oracle.get_number_of_weights()
        g = self.oracle.full_gradient(x) + lmb*x
        
        
        return x - np.linalg.inv(self.H).dot(g) 
    
    def method(self, x0, max_iter = 10, tol=10**(-15), init_cost=True, verbose=True):
        '''
        ---------------------------
        Implementation of NS method
        ---------------------------
        input:
        x0 - initial model weights
        max_iter - maximum number of steps of the method
        tol - desired tolerance of the solution
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        iterates - numpy array containing distances from current point to the solution
        bits - numpy array containing transmitted bits by one node to the server
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        n = self.oracle.get_number_of_nodes()
        func_value = []
        iterates = []
        bits = []
        
        global_bit = 1
        bits.append(global_bit)
        iterates.append(np.linalg.norm(x0-x_opt))
        func_value.append(self.oracle.function_value(x0))
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(np.linalg.norm(x0-x_opt))
            func_value.append(self.oracle.function_value(x0))
        
        if verbose:
            print(func_value[-1])

      
            
        n_steps = 0
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += 32*d
            bits.append(global_bit)
            x0 = self.step(x0)

            func_value.append(self.oracle.function_value(x0))
            iterates.append(np.linalg.norm(x0-x_opt))
            if verbose:
                print(func_value[-1])
            

        return np.array(func_value), np.array(bits), np.array(iterates)
    
###########################################################################    
class NL1:
    def __init__(self, oracle):
        '''
        -------------------------------------------------------------
        This class is created to simulate NEWTON-LEARN 1 (NL1) method
        -------------------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, H, max_iter=100, k=1, eta=None, tol=10**(-14), init_cost=True,\
               verbose=True):
        '''
        ----------------------------
        Implementation of NL1 method
        ----------------------------
        input:
        x - initial model weightsn
        H - list of vectors h_i^0
        max_iter - maximum number of iterations of the method
        k - the parameter of Rand-K compression operator
        eta - stepsize for update of vectors h_i's
        (if eta is None, then eta is set as k/m)
        tol - desired tolerance of the solution
        init_cost - if True, then the communication cost of initalization is inclued
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
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
                
        if eta is None:
            eta = k/m 
            
        f_opt = self.oracle.function_value(x_opt)    

        func_value = []
        iterates = []
        bits = []
        global_bit = 1
        bits.append(global_bit)
        
        iterates.append(np.linalg.norm(x-x_opt))
        func_value.append(self.oracle.function_value(x))
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
        
        if verbose:
            print(func_value[-1])

        

        n_steps = 0
        
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += 32*d + k*d*32 + 32*k
            
            global_grad = self.oracle.full_gradient(x)+lmb*x
            
            for i in range(n):
                H_old[i] = H_new[i]
                h = random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                H_new[i] = positive_part(H_old[i] + eta*h)
            
            D = - np.linalg.solve(B + lmb*np.eye(d), global_grad)
            x = x + D

            bits.append(global_bit)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
            
            if verbose:
                print(func_value[-1])

            for i in range(n):
                for j in range(m):
                    l = i*m+j
                    B += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                
        return np.array(func_value), np.array(bits), np.array(iterates)
    
########################################################################  

class NL1_Bernoulli:
    def __init__(self, oracle):
        '''
        ---------------------------------------------------------------------------------------
        This class is created to simulate NEWTON-LEARN 1 (NL1) method with bernoulli compressor
        ---------------------------------------------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, H, p, max_iter=100, k=1, eta=None, init_cost=True, tol=10**(-14), verbose=True):
        '''
        ----------------------------
        Implementation of NL1 method
        ----------------------------
        input:
        x - initial model weights
        H - list of vectors h_i^0
        p - parameter of Bernoulli compressor
        k - the parameter of Rand-K compression operator
        max_iter - maximum number of iterations of the method
        eta - stepsize for update of vectors h_i's
        init_cost - if True, then the communication cost of initalization is inclued
        (if eta is None, then eta is set as kp/m)
        tol - desired tolerance of the solution
        init_cost - if True, then the communication cost of initalization is inclued
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
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
        
        global_bit = 1
        bits.append(global_bit)
        iterates.append(np.linalg.norm(x-x_opt))
        func_value.append(self.oracle.function_value(x))
        
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
        
        if verbose:
            print(func_value[-1])

        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += 32*d + 32*k*d + 32*k
            bits.append(global_bit)
            global_grad = self.oracle.full_gradient(x)
            for i in range(n):
                H_old[i] = H_new[i]
                if bernoulli.rvs(p):
                    h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                else:
                    h = np.zeros(m)
                H_new[i] = H_old[i] + h
                
            x = x - np.linalg.inv(B + lmb*np.eye(d)).dot(global_grad + lmb*x)
            
            
            iterates.append(x)
            func_value.append(self.oracle.function_value(x))

            if verbose:
                print(func_value[-1])

            for i in range(n):
                for j in range(m):
                    l = i*m+j
                    B += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                
        return np.array(func_value), np.array(iterates), np.array(bits)
    
######################################################################## 
class NL2:
    def __init__(self, oracle):
        '''
        -------------------------------------------------------------
        This class is created to simulate NEWTON-LEARN 2 (NL2) method
        -------------------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, H, gamma, max_iter=100, k=1, eta=None,tol=10**(-14), init_cost=True, verbose=True):
        '''
        ----------------------------
        Implementation of NL2 method
        ----------------------------
        input:
        x - initial model weightsn
        H - list of vectors h_i^0
        gamma - parameter of NL2
        max_iter - maximum number of iterations of the method
        k - the parameter of Rand-K compression operator
        eta - stepsize for update of vectors h_i's
        (if eta is None, then eta is set as k/m)
        tol - desired tolerance of the solution
        init_cost - if True, then the communication cost of initalization is inclued
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
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

        global_bit = 1
        bits.append(global_bit)
        iterates.append(np.linalg.norm(x-x_opt))
        func_value.append(self.oracle.function_value(x))
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
        
        if verbose:
            print(func_value[-1])
        
        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += 32*d+32+32*k*d+32*k
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
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
            
            for i in range(n):
                H_old[i] = H_new[i]
                h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                H_new[i] = H_old[i] + h

                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                    
            if verbose:
                print(func_value[-1])
            
            
        return np.array(func_value), np.array(bits), np.array(iterates)
    
#################################################################################################    
    
    
class NL2_Bernoulli:
    def __init__(self, oracle):
        '''
        -------------------------------------------------------------
        This class is created to simulate NEWTON-LEARN 2 (NL2) method
        with Bernoulli compressor
        -------------------------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, H, p, gamma, max_iter=100, k=1, eta=None,tol=10**(-14), init_cost=True, verbose=True):
        '''
        ----------------------------
        Implementation of NL2 method
        ----------------------------
        input:
        x - initial model weightsn
        H - list of vectors h_i^0
        p - parameter of Bernoulli compressor
        gamma - parameter of NL2
        max_iter - maximum number of iterations of the method
        k - the parameter of Rand-K compression operator
        eta - stepsize for update of vectors h_i's
        (if eta is None, then eta is set as k/m)
        tol - desired tolerance of the solution
        init_cost - if True, then the communication cost of initalization is inclued
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
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

        global_bit = 1
        bits.append(global_bit)
        iterates.append(np.linalg.norm(x-x_opt))
        func_value.append(self.oracle.function_value(x))
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
        
        if verbose:
            print(func_value[-1])
            
            
        n_steps = 0
        while func_value[-1] - f_opt > tol and n_steps <= max_iter:
            n_steps += 1
            global_bit += 32*d + 32 + 32*d*k + 32*k
            
            
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
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
            
            for i in range(n):
                H_old[i] = H_new[i]
                if bernoulli.rvs(p):
                    h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                else:
                    h = np.zeros(m)
                H_new[i] = H_old[i] + h
                
                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                    
            bits.append(global_bit)
            
            if verbose:
                print(func_value[-1])
            
            
            
        return np.array(func_value), np.array(bits), np.array(iterates)
    

############################################################################################################
class CNL:
    def __init__(self, oracle):
        '''
        -----------------------------------------------------------------
        This class is created to simulate CUBIC-NEWTON-LEARN (CNL) method
        -----------------------------------------------------------------
        '''
        self.oracle = oracle
        
    def subproblem_solver(self, g, H, M, tol=10**(-15), max_iter=10000):
        '''
        -------------------------------------------------------
        Subproblem solver for <g,s> + 0.5*s^T*H*s + M/6*||s||^3
        -------------------------------------------------------
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
        ------------------------
        return: max_ij{\|a_ij\|}
        ------------------------
        '''
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        R = 0
        for i in range(N):
            R = max(R, np.linalg.norm(self.oracle.A[i]))
            
        return R
        
    def method(self, x, H, nu, gamma, max_iter = 5000, k=1, eta=None, tol=10**(-14), init_cost=True, verbose=True):
        '''
        ----------------------------
        Implementation of CNL method
        ----------------------------
        input:
        x - initial model weightsn
        H - list of vectors h_i^0
        nu, gamma - parameteres of CNL
        max_iter - maximum number of iterations of the method
        k - the parameter of Rand-K compression operator
        eta - stepsize for update of vectors h_i's
        (if eta is None, then eta is set as k/m)
        tol - desired tolerance of the solution
        init_cost - if True, then the communication cost of initalization is inclued
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()


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
        bits = []

        global_bit = 1
        bits.append(global_bit)
        iterates.append(np.linalg.norm(x-x_opt))
        func_value.append(self.oracle.function_value(x))
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
        
        if verbose:
            print(func_value[-1])
            

        n_steps = 0

        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += 32*d+32+32*k*d+32*k
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
            
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
            

            
            for i in range(n):
                H_old[i] = H_new[i]
                h = eta*random_k(self.oracle.alphas(x, i) - H_old[i], k = k)
                H_new[i] = H_old[i] + h

                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                    
            if verbose:
                print(func_value[-1])

        return np.array(func_value), np.array(bits), np.array(iterates)
        
        
        
        
############################################################################################################
class CNL_Bernoulli:
    def __init__(self, oracle):
        '''
        -----------------------------------------------------------------
        This class is created to simulate CUBIC-NEWTON-LEARN (CNL) method
        with Bernoulli compressor
        -----------------------------------------------------------------
        '''
        self.oracle = oracle
        
    def subproblem_solver(self, g, H, M, tol=10**(-15), max_iter=100):
        '''
        -------------------------------------------------------
        Subproblem solver for <g,s> + 0.5*s^T*H*s + M/6*||s||^3
        -------------------------------------------------------
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
        ------------------------
        return: max_ij{\|a_ij\|}
        ------------------------
        '''
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        R = 0
        for i in range(N):
            R = max(R, np.linalg.norm(self.oracle.A[i]))
            
        return R
        
    def method(self, x, H, nu, gamma, p = 1/2, max_iter = 5000, k=1, eta=None, tol=10**(-14), init_cost=True, verbose=True):
        '''
        ----------------------------
        Implementation of CNL method
        ----------------------------
        input:
        x - initial model weightsn
        H - list of vectors h_i^0
        nu, gamma - parameteres of CNL
        p - parameter of Bernoulli compressor
        max_iter - maximum number of iterations of the method
        k - the parameter of Rand-K compression operator
        eta - stepsize for update of vectors h_i's
        (if eta is None, then eta is set as k/m)
        tol - desired tolerance of the solution
        init_cost - if True, then the communication cost of initalization is inclued
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
        iterates - numpy array containing distances from current point to the solution
        '''
        x_opt = self.oracle.get_optimum()
        d = self.oracle.get_number_of_weights()
        lmb = self.oracle.get_reg_coef()
        n = self.oracle.get_number_of_nodes()
        m = self.oracle.get_number_of_local_data_points()
        N = n*m
        H_new = H.copy()
        H_old = H.copy()
        

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
        bits = []

        global_bit = 1
        bits.append(global_bit)
        iterates.append(np.linalg.norm(x-x_opt))
        func_value.append(self.oracle.function_value(x))
        
        if init_cost:
            global_bit = 32*d*(d+1)//2
            bits.append(global_bit)
            iterates.append(np.linalg.norm(x-x_opt))
            func_value.append(self.oracle.function_value(x))
        
        if verbose:
            print(func_value[-1])
            
            
        n_steps = 0
        while func_value[-1] - self.oracle.function_value(x_opt) > tol and n_steps <= max_iter:
            n_steps += 1
            
            global_bit += 32*d + 32*k*d + 32 + 32*k
            
            
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
                else:
                    h = np.zeros(m)
                H_new[i] = H_old[i] + h
                
                for j in range(m):
                    l = i*m+j
                    A_k += (H_new[i][j]-H_old[i][j])*self.oracle.A[l].reshape((d,1)).dot(self.oracle.A[l].reshape(1,d))/N
                    
            bits.append(global_bit)
            
            if verbose:
                print(func_value[-1])
                

        return np.array(func_value), np.array(bits), np.array(iterates)
    
###############################################################################


class DINGO:
    
    def __init__(self, oracle):
        '''
        ----------------------------------------------
        This class is created to simulate DINGO method
        ----------------------------------------------
        '''
        self.oracle = oracle
        
    def method(self, x, max_iter=200, tol=1e-15, phi=1e-6, theta=1e-4, rho=1e-4, verbose=True):
        '''
        -------------------------
        Implementation of DINGO method
        -------------------------
        
        input:
        x - initial point
        max_iter - maximum iterations of the method
        tol - desired tolerance of the solution
        phi, theta, rho - parameters of DINGO
        verbose - if True, then function values in each iteration are printed
        
        return:
        func_value - numpy array containing function value in each iteration of the method
        bits - numpy array containing transmitted bits by one node to the server
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
        global_bit = 1
        bits.append(global_bit)
        func_value.append(self.oracle.function_value(x))
        n_steps = 0

        if verbose:
            print(func_value[-1])
        delta = self.oracle.function_value(x_opt+np.ones(d)*0.1) - f_opt
        n_steps = 0
        while func_value[-1] - f_opt > tol and max_iter >= n_steps:
            
            
            global_bit += 32*d # local gradient to the master
            global_bit += 32*d # global gradient to the node
            global_bit += 3*32*d # 3 types of steps
            
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
                H_inv.append(np.linalg.pinv(B).dot(g))
                h_inv += 1/n*H_inv[i]
                H = np.vstack((B, phi*np.eye(d)))
                G = np.vstack((g.reshape(d,1), np.zeros((d,1))))
                H_hat.append(np.linalg.pinv(H).dot(G))
                h_hat += 1/n*H_hat[i].squeeze()


            if h_i.dot(h_inv) >= theta*g_norm:
                p = -h_inv
            elif h_i.dot(h_hat) >= theta*g_norm:
                p = -h_hat
            else:
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
                        
            global_bit += 32*2*d # H_t*g_t to the node and p_i to the master
            a = 1
            g_next = self.oracle.full_gradient(x+a*p)+lmb*(x+a*p)
            g_next_norm = np.linalg.norm(g_next)**2
            
            while g_next_norm > g_norm + 2*a*rho*p.dot(h_i):
                global_bit += 32*2*d
                a /= 2
                g_next = self.oracle.full_gradient(x+a*p)+lmb*(x+a*p)
                g_next_norm = np.linalg.norm(g_next)**2
               
            if n_steps < 5:
                a = min(1e-2, a)
            a = max(a, 2**(-10))
            x = x + a*p
            func_value.append(self.oracle.function_value(x))
            bits.append(global_bit)
            
            if verbose:
                print(func_value[-1])

            
        return np.array(func_value), np.array(bits)

####################################################################

    
    
    
def dcgd(X, y, w, arg, f_opt, tol=1e-15, verbose=True):
    '''
    -----------------------------
    Implementation of DCGD method
    -----------------------------
    X - data matrix
    y - labels vectors 
    w - initial point 
    arg - class containing all parameters of method and comressor
    f_opt - optimal function value
    tol - desired tolerance of the solution
    verbose - if True, then function values in each iteration are printed

    return:
    loss - numpy array containing function value in each iteration of the method
    com_bits - numpy array containing transmitted bits by one node to the server
    '''
    
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
    if verbose:
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
        if verbose:
            if k % 1000 == 0:
                print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits


def diana(X, y, w, arg, f_opt, tol=1e-15, verbose=True):
    '''
    -------------------------
    Implementation of DIANA method
    -------------------------
    X - data matrix
    y - labels vectors 
    w - initial point 
    arg - class containing all parameters of method and comressor
    f_opt - optimal function value
    tol - desired tolerance of the solution
    verbose - if True, then function values in each iteration are printed

    return:
    loss - numpy array containing function value in each iteration of the method
    com_bits - numpy array containing transmitted bits by one node to the server
    '''
    alg = 'DIANA'
    dim = X.shape[1]
    
    omega = compute_omega(dim, arg)
    arg.alpha = 1 / (1 + omega)
    arg.eta = min(arg.alpha / (2 * arg.lamda), 2 / ((arg.L + arg.lamda) * (1 + 6 * omega / arg.node)))
    if verbose:
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
    if verbose:
        print('at iteration 0', 'loss =', loss_0)
    loss.append(loss_0)
    
    com_bits = [1]
    bits = 1
    
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
        if verbose:
            if k % 1000 == 0:
                print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits


def adiana(X, y, w, arg, f_opt, tol=1e-15, verbose=True):
    '''
    -------------------------
    Implementation of DIANA method
    -------------------------
    X - data matrix
    y - labels vectors 
    w - initial point 
    arg - class containing all parameters of method and comressor
    f_opt - optimal function value
    tol - desired tolerance of the solution
    verbose - if True, then function values in each iteration are printed

    return:
    loss - numpy array containing function value in each iteration of the method
    com_bits - numpy array containing transmitted bits by one node to the server
    '''
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
    
    if verbose:
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

    
    if verbose:
        print('at iteration 0', 'loss =', loss_0)
    loss.append(loss_0)
    
    com_bits = [1]
    bits = 1
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
        if verbose:
            if k % 1000 == 0:
                print('at iteration', k + 1, ' loss =', loss_k)
    loss = np.array(loss)
    com_bits = np.array(com_bits)
    return loss, com_bits