import numpy as np
import matplotlib.pyplot as plt
from layer import InnerLayer, SoftmaxLayer
from activations import Activations
from loss_functions import LossFunctions
from utils import sample_indices


class Network():

    def __init__(self, layers = list(), learning_rate=0.001, loss_function = LossFunctions.mse, regularization=None, pen_rate=0.0):
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.train_loss = list() #saving loss for displaying
        self.valid_loss = list()
        self.regularization=regularization
        self.pen_rate = pen_rate
        if self.layers[0].input_size is None:
            raise Exception("The input size of the input layer mast be specified!")
        for i in range(1, len(self.layers)):
            if self.layers[i].input_size is None:
                self.layers[i].set_input_size(self.layers[i-1].units)
        
        for layer in self.layers:
            if isinstance(layer, InnerLayer) and layer.learning_rate is None:
                layer.learning_rate = self.learning_rate
        

    def forward_pass(self, train_cases, targets, cash_info=True):
        train_cases = np.array(train_cases)
        inputs = train_cases
        self.inputs=np.array(inputs)
        for layer in self.layers:
            inputs = layer.forward_pass(inputs, cash_info)
        if cash_info:    
            self.predictions = inputs
            self.targets = np.array(targets)

       
    def backward_pass(self):
        # 1. Compute the initial Jacobian representing the derivative of the loss wrt the outputs, cached on forward stage
        self.J_L_Z = self.loss_function(self.predictions, self.targets).derivative() # jacobian for all the cases in the minibatch if specified, 

        J_L_N = self.J_L_Z
        for i in range(len(self.layers)-1, -1,-1):
            J_L_N = self.layers[i].backward_pass(J_L_N, regularization=self.regularization, pen_rate=self.pen_rate)
        
    
    def train(self, train_set, valid_set = None, epochs=4, batch_size=4, verbose = False):
        cases = train_set[0]
        targets = train_set[1]
        if valid_set is not None:
            self.valid_cases = valid_set[0]
            self.valid_targets = valid_set[1]
        self.train_targets = targets
        self.all_train_cases = cases
        if batch_size > len(cases):
            batch_size = len(cases)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            running_loss = 0.0
            i = 0
            while i < len(cases):
                batch_cases = cases[i:i+batch_size]
                batch_targets = targets[i:i+batch_size]
                
                self.forward_pass(batch_cases, batch_targets)
                self.backward_pass()

                ## calculating loss for displaing
                batch_predictions = self.predict(batch_cases)
                running_loss += np.mean(self.loss_function(batch_predictions, self.train_targets[i:i+batch_size]).func() )
                #self.train_loss.append(np.mean(loss))
                i += batch_size
            batches = len(cases)/batch_size
            self.train_loss.append(running_loss*1.0/batches)
            
            if valid_set is not None:
                all_valid_predictions = self.predict(self.valid_cases)
                v_loss = self.loss_function(all_valid_predictions, self.valid_targets).func() 
                self.valid_loss.append(np.mean(v_loss))
            
            print(f"Train loss: {self.train_loss[-1]}, validation loss: {self.valid_loss[-1]}")
        
        if verbose:
            f =  open('network_output.txt', 'w')
            f.seek(0)
            f.write('Netwotk inputs: \n')
            for case in train_set[0]:
                f.write(str(case)+"\n")
            f.write('Network outputs:\n')
            for p in self.predict(self.all_train_cases):
                f.write(str(p)+"\n")
            f.write('Target values:\n')
            for t in self.train_targets:
                f.write(str(t)+"\n")
            f.write("Loss:")
            f.write(str(self.train_loss[-1])+"\n")
            f.write("Weights and biases\n")
            i = 1
            for i in range(len(self.layers)):
                f.write("\nLayer "+str(i)+"\n")
                i += 1
                if isinstance(self.layers[i-1], SoftmaxLayer):
                    f.write("\nSoftmax layer, no weights and biases\n")
                else:
                    f.write("\nWeights:\n")
                    f.write(str(self.layers[i-1].weights))
                    f.write("\nBiases:\n")
                    f.write(str(self.layers[i-1].biases))
            f.close()


    def predict(self, test_cases):
        inputs = np.array(test_cases)
        for layer in self.layers:
            inputs = layer.forward_pass(inputs, cash_info=False)
        return inputs

    def show_loss_graph(self):
        fig, ax = plt.subplots()
        graph1, = ax.plot(self.train_loss, label="Train Loss")
        legend1 = ax.legend(handles =[graph1],  loc ='upper center')
        ax.add_artist(legend1)

        if len(self.valid_loss) > 0:
            graph2, = ax.plot(self.valid_loss, label = "Validation Loss")
            legend2 = ax.legend(handles =[graph2],  loc ='lower center')
            ax.add_artist(legend2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def calculate_loss(self, test_set, show=False):
        cases = test_set[0]
        targets = test_set[1]
        predictions = self.predict(cases)
        loss = self.loss_function(predictions, targets).func() 
        if show:
            print(f"Loss for the test set is {np.mean(loss):.2f}")
        return loss

def main():
    network = Network(layers=[InnerLayer(input_size=3, units=2, activation=Activations.sigmoid), InnerLayer(input_size=2, units=3, activation=Activations.sigmoid) ])
    network1 = Network(layers=[InnerLayer(input_size=2, units=2, activation=Activations.sigmoid), InnerLayer(input_size=2, units=1, activation=Activations.linear) ],
    loss_function=LossFunctions.mse)
    network2 = Network(layers=[InnerLayer(input_size=2, units=2, activation=Activations.tanh), InnerLayer(input_size=2, units=1, activation=Activations.linear) ],
    loss_function=LossFunctions.mse) # works best for
    network3 = Network(layers=[InnerLayer(input_size=2, units=2, activation=Activations.relu), InnerLayer(input_size=2, units=1, activation=Activations.linear) ],
    loss_function=LossFunctions.mse)
   # network2.train(cases=[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]], targets=[[1.0], [0.0], [0.0], [1.0]], epochs=100000, batch_size=2)

    network4 = Network(layers=[InnerLayer( input_size=2,units=2, activation=Activations.tanh), InnerLayer(units=1, activation=Activations.linear)], learning_rate=0.01, loss_function=LossFunctions.mse)
    #network4.train(cases=[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]], targets=[[1.0], [0.0], [0.0], [1.0]], epochs=100000, batch_size=2)

    network5 = Network(layers=[InnerLayer( input_size=2,units=3, activation=Activations.sigmoid), InnerLayer(units=3, activation=Activations.linear),
    SoftmaxLayer()],
     learning_rate=0.01, loss_function=LossFunctions.cross_entropy) 
    network5.train(cases=[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]], targets=[[1.0, 0.0,0.0], [0.0, 1.0,0.0], [0.0, 1.0,0.0], [0.0, 0.0, 1.0]], epochs=10000, batch_size=4)
    print("----predictions----")   
    print(network5.predict([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0],[0.0, 0.0]]))

    #print("biases")
    #print(network4.layers[0].biases)
    #print(network4.layers[1].biases)

    #print("weights")
    #print(network4.layers[0].weights)
    #print(network4.layers[1].weights)
if __name__ == '__main__':
    main()