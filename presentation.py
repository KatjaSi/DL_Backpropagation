import numpy as np
from config_parser import NetworkParser, DataParser
from data_generator import DataGenerator
from vizualizer import Visualizer


def main():
    #cp = NetworkParser("config_2.ini") # main one
    #cp = NetworkParser("no_hidden_layers_network.ini")
    #cp = NetworkParser("many_layers.ini")
    cp = NetworkParser("many_layers2.ini") # config_2 is the best or 3
    dp = DataParser("data3.ini") 
    network = cp.parse_network()
    train, valid, test = dp.parse_data() 
    network.train(train_set=train[:2], valid_set = valid[:2], epochs=20, batch_size=20, verbose=True)
    # three sets of images: train, validation and test. Each set is a 5-item tuple: (images, targets, labels, 2d-image-dimensions, flat)
    visualizer = Visualizer()
    #visualizer.show_n_random(train[0],10, flatened=True)
    #print(valid[1][:7])
    #print("\n")
    #print(network.predict(valid[0])[:7])
    network.show_loss_graph()
    #test_predictions = network.predict(test[0])
    network.calculate_loss(test[:2], show=True)


if __name__ == '__main__':
    main()