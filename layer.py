import numpy as np
from utils import init_weights
from activations import Activations

class Layer():
    """
    Base class from which other Layer classes inherit 
    """
    def __init__(self):
        self.output = None
        self.input_size = None

    def set_input_size(self, input_size):
        self.input_size = input_size

    def forward_pass(self, inputs):
        pass

    def backward_pass(self):
        pass

class InnerLayer(Layer):
    
    def __init__(self, input_size=None, units=32, activation=Activations.linear, learning_rate=None, wr=(-0.1, 0.1), br=(0.0, 0.0)):
        self.input_size = input_size
        self.units = units # the number of neurons in the layer
        self.wr = wr
        self.biases = init_weights((units), br[0], br[1])
        if input_size is not None:
            self.set_input_size(input_size)
        self.activation = activation
        self.learning_rate = learning_rate

    def set_input_size(self, input_size):
        super().set_input_size(input_size)
        self.weights = init_weights((input_size, self.units), self.wr[0], self.wr[1])

    def forward_pass(self, inputs,cash_info=True):
        self.inputs = inputs
        output =  self.activation.func(np.dot(inputs, self.weights)+ self.biases)
        if cash_info:
            self.output = output
        return output

    def backward_pass(self, J_L_N, regularization=None, pen_rate=0.0):#, case): # case is the number of case in the batch
        """
        J_sum_z = self.activation.derivative(self.output[case])
        J_y_z = np.dot(J_sum_z, self.weights.T) # connecting y and z layers
        J_hat_w_z = np.outer(self.inputs[case], np.diag(J_sum_z)) # connecting outputs to the incoming wrights
        J_w_L = J_L_N*J_hat_w_z # same dimentionality as weights
        # onlu sum when all case from th minibatch passed
        #self.weights -= 0.01*J_w_L
        self.d_weights -= 0.01*J_w_L
        # becase d output/d_bias = identity matrix, so 
        #self.biases -= 0.01*np.dot(J_L_N, 1)
        self.d_biases -= 0.01*np.dot(J_L_N, 1)
        J_y_L = np.dot(J_L_N, J_y_z) # to be passed
        return J_y_L
        """
        #### matricer
        
        J_sum_z =  self.activation.derivative_batch(self.output)#np.apply_along_axis(self.activation.derivative, -1, self.output)
        J_y_z = np.dot(J_sum_z, self.weights.T) # connecting y and z layers 
        #J_hat_w_z = [np.outer(self.inputs[i], np.diag(J_sum_z[i])) for i in range(len(self.inputs))] # connecting outputs to the incoming wrights
        J_hat_w_z = np.einsum('ik,ijj->ikj', self.inputs, J_sum_z)
        J_w_L = np.sum(np.einsum('ji,jki->jki', J_L_N, J_hat_w_z),axis=0)
        #J_w_L = np.sum([J_L_N[i]*J_hat_w_z[i] for i in range(len(self.inputs))], axis=0) # same dimentionality as weights
        # onlu sum when all case from th minibatch passed
        if regularization is not None:
            self.regularize(regularization, pen_rate)
        self.weights -= self.learning_rate*J_w_L
        # becase d output/d_bias = identity matrix, so 
        self.biases -= self.learning_rate*np.sum(np.dot(J_L_N,1),axis=0) 
       # J_y_L = np.dot(J_L_N, J_y_z) # to be passed 
        #J_y_L = np.sum(np.dot(J_L_N, J_y_z), axis = 1) 
        J_y_L = np.einsum('ki,kij->kj', J_L_N, J_y_z)
        return J_y_L
        
    def regularize(self, regularization, pen_rate):
        if regularization == "L2":
            self.weights -= pen_rate*self.weights
            self.biases -= pen_rate*self.biases
        if regularization == "L1":
            self.weights -= pen_rate*np.sign(self.weights)
            self.biases -= pen_rate*np.sign(self.biases)
        

class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()
        self.activation = Activations.soft_max
        self.learning_rate = None

    def forward_pass(self, inputs,cash_info=True):
        output =  self.activation.func(inputs)
        if cash_info:
            self.output = output
        return output

    def backward_pass(self,  J_L_N, regularization, pen_rate):
        # regularization and pen_rate is not used
        J_sum_z =  self.activation.derivative_batch(self.output)
        J_y_L = np.einsum('kij,kj->ki', J_sum_z, J_L_N) #[np.dot(J_sum_z[i], J_L_N[i]) for i in range(len(self.output))]
        return J_y_L

def main():
    layer = InnerLayer(2, 3, activation=Activations.relu) # input from layer with 2 neurons to our layer with 3 neurons
    inputs = np.array([0.01,0.001])
    print(layer.forward_pass(inputs))
    

if __name__ == '__main__':
    main()