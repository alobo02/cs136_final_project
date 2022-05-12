import numpy as np
import random
import warnings
from numpy.random import RandomState


class MAPEstimator():
    """
    Maximum A-Posteriori Estimator for musk probabilities

    Attributes
    ----------
    w_D       : D-dimensional vector of reals
                Defines weight vector
    prior     : string
                Defines prior that will be used options: {'trivial','sas'}
                - 'sas' : stands for 'spike-and-slab'
    solver    : string
                The type of gradient descent used {'fo', 'so'}
                - 'fo' : first-order schochastic gradient descent
                - 'so' : second-order schochastic gradient descent 
    alpha     : float, must be greater than 0
                Defines precision parameter of the multivariate Gaussian prior for the weight vector or the "spike" multivariate Gaussian on the GMM prior for the weight vector
    beta      : float, must be greater than 0 (optional)
                Defines precision parameter of the "slab" Gaussian on the GMM prior for the weight vector
    max_iter  : integer
                Defines max number of iterations model can take to converge on fit()
    step_size : float must be greater than 0
                The step size of the gradient descent algorithm
    max_iter  : int greater than 0
                Maximum number of iterations that the gradient descent algorithm will take
            
            
    Examples
    # TODO : Update this example or delete it
    --------
    >>> word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus']
    >>> mapEst = MAPEstimator(Vocabulary(word_list), alpha=2.0)
    >>> mapEst.fit(word_list)
    >>> np.allclose(mapEst.predict_proba('dinosaur'), 3.0/7.0)
    True

    >>> mapEst.predict_proba('never_seen-before')
    Traceback (most recent call last):
    ...
    KeyError: 'Word never_seen-before not in the vocabulary'
    """

    def __init__(self, w_D, prior='trivial', solver='fo', step_size_type = 'universal', alpha=1.0, beta=None, max_iter=30000, tol=1e-4, step_size=1.0, c = 0):
        ## Weight vector
        self.w_D = w_D
        self.c = c
        self.w = np.hstack((self.c, self.w_D))
        
        ## Solver Parameters
        self.prior = prior
        self.solver = solver
        
        ## Hyperparamters
        self.alpha = float(alpha)
        if prior != 'trivial':
            self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.step_size_type = step_size_type
        
        ## Other
        self.iteration_count = 0
        self.loss_array = []

    def fit(self, train_X, train_y):
        ''' 
        Fit this estimator to provided training data with first/second order stochastic gradient descent

        Args
        ----
        train_X : NxD array
            Each entry is a D-dimensional vector representing a training example
        train_y : Nx1 array
            Each entry is corresponding output class for training data
            
                    
        Returns
        -------
        None. Internal attributes updated.

        Post Condition
        --------------
        Attributes will updated based on provided word list
        * The 1D array count_V is set to the count of each vocabulary word
        * The integer total_count is set to the total length of the word list
        '''
        if self.solver == 'fo':
        
            self.iteration_count = 0
            example_num = 0
            num_examples = len(train_X)
            self.loss_array = []
            
            X = np.hstack((np.ones((num_examples,1)), train_X))
            
            if self.prior == 'trivial':
                diff_L = np.inf
                L = np.inf
                
                prng = RandomState(136)
                
                while(self.iteration_count <= self.max_iter) and (diff_L > self.tol):
                    L_prev = L
                    L = (-(train_y[:, np.newaxis].T @ np.log(sigmoid(X @ self.w)) + (1 - train_y[:, np.newaxis]).T @ np.log(1 - sigmoid(X @ self.w))) + 0.5 * self.alpha * np.dot(self.w, self.w)) / num_examples
                    
                    example_num = prng.randint(0, num_examples-1)
                    
                    y = sigmoid(X @ self.w)
                    g = X.T @ (y - train_y) + self.alpha * self.w
                    
                    ## Old (stochastic)
                    #h_x = np.dot(self.w_D, train_X[example_num]) + self.c
                    #sig = 1 / (1 + np.exp(-h_x))
                    diff_L = np.abs(L_prev - L)
                    self.loss_array.append(L)
                    if self.step_size_type == 'universal':
                        self.w = self.w - self.step_size * g
                        #self.c = self.c - self.step_size * (sig - train_y[example_num])
                    else:
                        self.w_D = self.w_D - (self.step_size + train_y[example_num]*self.step_size)* ((sig - train_y[example_num]) * train_X[example_num] + self.alpha * self.w_D)
                        self.c = self.c - (self.step_size + train_y[example_num]*self.step_size) * (sig - train_y[example_num])
                    self.iteration_count += 1
                if self.iteration_count >= self.max_iter:
                    warnings.warn("Maximum iterations reached")

            #else:
                #TODO: Code up the SGD for the sas prior
            
        else: #elseif self.solver == 'fo':
                
            self.iteration_count = 0
            example_num = 0
            num_examples = len(train_X)
            self.loss_array = []
            
            X = np.hstack((np.ones((num_examples,1)), train_X))

            if self.prior == 'trivial':
                diff_L = np.inf
                L = np.inf

                prng = RandomState(136)

                while(self.iteration_count <= self.max_iter) and (diff_L > self.tol):
                    L_prev = L
                    #L = -(train_y[:, np.newaxis].T @ np.log(sigmoid(train_X @ self.w_D + self.c)) + (1 - train_y[:, np.newaxis]).T @ np.log(1 - sigmoid(train_X @ self.w_D + self.c))) + 0.5 * self.alpha * np.dot(self.w_D, self.w_D)
                    #L = -(train_y[:, np.newaxis].T @ np.log(sigmoid(train_X @ self.w_D + self.c)) + (1 - train_y[:, np.newaxis]).T @ np.log(1 - sigmoid(train_X @ self.w_D + self.c)))
                    L = (-(train_y[:, np.newaxis].T @ np.log(sigmoid(X @ self.w)) + (1 - train_y[:, np.newaxis]).T @ np.log(1 - sigmoid(X @ self.w))) + 0.5 * self.alpha * np.dot(self.w, self.w)) / num_examples
                    #print(L)

                # TODO : spit out a warning if the max_iter is reached
                    #print('iteration: %i' % self.iteration_count)
                    example_num = prng.randint(0, num_examples-1)
                    
                    #sig = sigmoid(np.dot(self.w_D, train_X[example_num]) + self.c)
                    #r_n = sig * (1 - sig)
                    
                    y = sigmoid(X @ self.w)
                    g = X.T @ (y - train_y) + self.alpha * self.w
                    R = np.diag(y*(1-y))
                    H = X.T @ R @ X + self.alpha

                    #H_w = r_n * (train_X[example_num][:, np.newaxis] @ train_X[example_num][:, np.newaxis].T) + self.alpha
                    
                    
                    #H_w = r_n * (train_X[example_num][:, np.newaxis] @ train_X[example_num][:, np.newaxis].T)
                    #print('H_w inverse shape:' + str(np.linalg.inv(H_w).shape))
                    #g_w = (sig - train_y[example_num]) * train_X[example_num] + (self.alpha * self.w_D)
                    #g_w = (sig - train_y[example_num]) * train_X[example_num]
                    #print('g_w shape:' + str(g_w[:, np.newaxis].shape))
                    

                    diff_L = np.abs(L_prev - L)
                    self.loss_array.append(L)

                    if self.step_size_type == 'universal':
                        self.w = self.w - self.step_size * np.linalg.inv(H) @ g
                        #self.w_D = self.w_D - self.step_size * (np.linalg.inv(H_w) @ g_w)
                        #self.c = self.c - self.step_size * (1/r_n) * (sigmoid(np.dot(self.w_D, train_X[example_num]) + self.c) - train_y[example_num])
                        #self.c = self.c - self.step_size * (sig - train_y[example_num])
                    else:
                        self.w_D = self.w_D - (self.step_size + train_y[example_num]*self.step_size) * (np.linalg.inv(H_w) @ g_w)
                        self.c = self.c - (self.step_size + train_y[example_num]*self.step_size) * (1/r_n) * (sigmoid(np.dot(self.w_D, train_X[example_num]) + self.c) - train_y[example_num])
                    self.iteration_count += 1

                if self.iteration_count >= self.max_iter:
                    warnings.warn("Maximum iterations reached")

            #else:
                #TODO: Code up the SGD for the sas prior

    def predict_proba(self, test_X):
        ''' 
        Predict probability of a given set of feature vectors under this model

        Args
        ----
        test_X : NxD vector
            Examples for which probability will be predicted

        Returns
        -------
        proba : float between 0 and 1

        Throws
        ------
        ValueError if hyperparameters do not allow MAP estimation
        KeyError if the provided word is not in the vocabulary
        '''
        
        num_examples = len(test_X)
        X = np.hstack((np.ones((num_examples,1)), test_X))
        
        lin_preds = np.matmul(X, self.w)
        
        sig_preds = 1 / (1 + np.exp(-lin_preds))
        
        return sig_preds

    def score(self, test_X, test_y, threshold):
        ''' Compute the average log probability of words in provided list

        Args
        ----
        test_X : NxD array
            Each entry is a D-dimensional vector representing a test example
        test_y : Nx1 array
            Each entry is corresponding output class for test data

        Returns
        -------
        avg_log_proba : float between (-np.inf, 0.0)
        '''
        correct_count = 0
        num_examples = len(test_y)
        
        pred_proba = self.predict_proba(test_X)
        pred_class = pred_proba > threshold
        is_correct = pred_class == test_y
        
        return np.sum(is_correct) / num_examples
    
    
def sigmoid(X):
    return 1 / (1 + np.exp(-X))