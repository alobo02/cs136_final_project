import numpy as np

class MAPEstimator():
    """
    Maximum A-Posteriori Estimator for musk probabilities

    Attributes
    ----------
    w_D   : D-dimensional vector of reals
            Defines weight vector
    alpha : float, must be greater than 0
            Defines precision parameter of the multivariate Gaussian prior on the weight vector
    iteration_count: integer
            Defines number of iterations it took for model to converge on last call of fit()

    Examples
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

    def __init__(self, w_D, alpha=1.0):
        self.w_D = w_D
        self.alpha = float(alpha)
        self.iteration_count = 0
        

    def fit(self, train_X, train_y, step_size = 1.0, max_iter = 100):
        ''' Fit this estimator to provided training data with first order stochastic gradient descent

        Args
        ----
        train_X : NxD array
            Each entry is a D-dimensional vector representing a training example
        train_y : Nx1 array
            Each entry is corresponding output class for training data
        step_size: float must be greater than 0
            The step size of the gradient descent algorithm
        max_iter: int greater than 0
            Maximum number of iterations that the gradient descent algorithm will take
                    
        Returns
        -------
        None. Internal attributes updated.

        Post Condition
        --------------
        Attributes will updated based on provided word list
        * The 1D array count_V is set to the count of each vocabulary word
        * The integer total_count is set to the total length of the word list
        '''
        self.iteration_count = 0
        
        example_num = 0
        num_examples = len(train_X)
        
        while(self.iteration_count <= max_iter):
            h_x = np.dot(self.w_D, train_X[example_num])
            sig = 1 / (1 + exp(-h_x))
            self.w_D = self.w_D - step_size * (sig - test_y[example_num]) * train_X[example_num] + self.alpha * self.w_D
            example_num += 1
            if example_num >= num_examples:
                example_num = 0
            self.iteration_count += 1

    def predict_proba(self, test_X):
        ''' Predict probability of a given set of feature vectors under this model

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

        # TODO calculate MAP estimate of the provided word
        
        lin_preds = np.matmul(test_X, self.w_D)
        
        sig_preds = 1 / (1 + exp(-lin_preds))
        
        return sig_preds

    def score(self, test_X, test_y):
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
        num_examples = len(test_X)
    
        for i in range(num_examples):
            pred_proba = self.predict_proba(test_X[i])
            if pred_proba < 0.5:
                pred_class = 0
            else:
                pred_class = 1
                
            if (pred_class == test_y[i]):
                correct_count += 1
            
        return correct_count / num_examples