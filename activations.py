import numpy as np 

class Activations():
    """
    All derivative for all calculations are calculated based on the output of the layer
    """

    class linear():
        func = lambda x: x
        
        def derivative(x):
            derivative = np.zeros((len(x),len(x)  ), float)
            np.fill_diagonal(derivative, 1.0)
            return derivative
        
        def derivative_batch(o):
            return np.apply_along_axis(Activations.linear.derivative, -1, o)
    
    class sigmoid():
        func = lambda x : 1/(1+np.e**(-x))
        def derivative(o):
            derivative = np.zeros((len(o),len(o)  ), float)
            diag = o *(1-o) # o is output of the layer, so o = sigmoid(x)
            np.fill_diagonal(derivative, diag)
            return derivative
            
        def derivative_batch(o):
            return np.apply_along_axis(Activations.sigmoid.derivative, -1, o)

    class tanh():
        func = lambda x : (np.e**x - np.e**(-x))/(np.e**x + np.e**(-x))
        def derivative(o): # derivative for one output
            derivative = np.zeros((len(o),len(o) ), float)
            diag = 1-o**2 # o is output of the layer, so o = tanh(x)
            np.fill_diagonal(derivative, diag)
            return derivative
        def derivative_batch(o):
            return np.apply_along_axis(Activations.tanh.derivative, -1, o)

    class relu():
        func = lambda x: np.maximum(np.zeros_like(x), x)

        def derivative(o):
            derivative = np.zeros((len(o),len(o) ), float)
            diag = np.where(o > 0, 1.0, 0.0) # o is output of the layer, so o = tanh(x)
            np.fill_diagonal(derivative, diag)
            return derivative
        
        def derivative_batch(o):
            return np.apply_along_axis(Activations.relu.derivative, -1, o)

    class soft_max():
        def func(x):
            sums = np.sum(np.e**x, axis=1)
            return np.e**x/sums.reshape(-1,1).repeat(x.shape[1], axis=1)

        def derivative(o):
            derivative = np.array([[-o[i]*o[j]  for j in range(len(o))] for i in range(len(o))])
            diag = np.array([o[i]-o[i]**2 for i in range(len(o))])
            np.fill_diagonal(derivative, diag)
            return derivative

        def derivative_batch(o):
            return np.apply_along_axis(Activations.soft_max.derivative, -1, o)
