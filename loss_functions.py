import numpy as np
import math

class LossFunctions(): # all loss func-s and there derivatives r calc-ed for the batch

    class mse():
    
        def __init__(self, predictions, targets): # predictions and targets are for a batch
            self.predictions = predictions
            self.targets = targets

        def func(self): 
            return np.square(np.subtract(self.targets,self.predictions)).mean(axis=1)

        def derivative(self):
            N = self.predictions.shape[1] # the number of output nodes 
            return 2*(self.predictions-self.targets)/N # slide 47, lecture 2/3

    class cross_entropy(): #TODO: add derivative
        """
        use when targets represent probability distributions
        """
        def __init__(self, predictions, targets):
            self.predictions = predictions
            self.targets = targets


        def func(self):
            small_vals =  np.arange(len(self.predictions),dtype=float)
            #small_vals.fill(1e-15)
            return -np.sum(self.targets*np.log2(self.predictions + 1e-15),axis=1)

        def derivative(self):
            predictions = np.where(abs(self.predictions-0.0)<1e-15, 1e-15, self.predictions)
            predictions = np.where(abs(self.predictions-1.0)<1e-15, 1-1e-15,predictions)
            return -self.targets/predictions


    class binary_cross_entropy():
        """
        use for single-value prediction where output = prob of class membership
        """
        def __init__(self, predictions, targets):
            self.predictions = predictions
            self.targets = targets


        def func(self):
            return LossFunctions.cross_entropy(self.predictions, self.targets).func() \
                +LossFunctions.cross_entropy(np.ones(len(self.predictions))-self.predictions, np.ones(len(self.targets))-self.targets).func()

        def derivative(self):
            predictions = np.where(self.predictions != 0, self.predictions, 1e-15)
            predictions = np.where(predictions != 1, predictions, 1-1e-15)
            result = -np.divide(self.targets,predictions) + np.divide(1-self.targets,1-predictions) 
            return  result