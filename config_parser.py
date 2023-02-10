from configparser import ConfigParser
from activations import Activations
from network import Network
from layer import InnerLayer, SoftmaxLayer
from loss_functions import LossFunctions
from data_generator import DataGenerator
from utils import str_to_bool

class NetworkParser(ConfigParser):

    def __init__(self, config_file):
        super().__init__()
        self.loss_functions = {
            "cross_entropy":LossFunctions.cross_entropy,
            "mse":LossFunctions.mse
        }
        self.activations = {
            "tanh":Activations.tanh,
            "sigmoid":Activations.sigmoid,
            "linear":Activations.linear,
            "relu":Activations.relu
        }
        self.cp = ConfigParser()
        self.cp.read(config_file)

    def parse_network(self):
        loss=self.cp["NETWORK_PARAMETERS"]["loss"]
        lr = float(self.cp["NETWORK_PARAMETERS"]["lr"])
        input_size = int(self.cp["INPUT"]["input_size"])
        layers=list()
        for section in self.cp.sections():
            if section[:5]=="LAYER":
                layers.append(self.__parse_layer__(section))
        layers[0].set_input_size(input_size)
        regularization=None
        pen_rate=0.001
        if "regularization" in self.cp["NETWORK_PARAMETERS"]:
            regularization = self.cp["NETWORK_PARAMETERS"]["regularization"]
        if "penalty_rate" in self.cp["NETWORK_PARAMETERS"]:
            pen_rate = float(self.cp["NETWORK_PARAMETERS"]["penalty_rate"])
        network = Network(layers=layers, learning_rate=lr, loss_function=self.loss_functions.get(loss), regularization=regularization, pen_rate=pen_rate)
        return network

    def __parse_layer__(self, section):
        if "type" in self.cp[section] and self.cp[section]["type"]=="softmax":
            return SoftmaxLayer()
        units = int(self.cp[section]["units"])
        lr =  self.cp[section]["lr"] if "lr" in self.cp[section] else None
        layer = InnerLayer(units=units, learning_rate = lr)
        if "act" in self.cp[section]:
            layer.activation = self.activations.get(self.cp[section]["act"])
        if "wr" in self.cp[section]:
            layer.wr = self.__parse_range__(self.cp[section]["wr"])
        if "br" in self.cp[section]:
            layer.wr = self.__parse_range__(self.cp[section]["br"])
        return layer

    def __parse_range__(self, range_str):
        range = tuple([float(x) for x in range_str[1:-1].split()])
        return range


class DataParser(ConfigParser):

    def __init__(self, config_file):
        super().__init__()
        self.cp = ConfigParser()
        self.cp.read(config_file)

    def parse_data(self):
        img_count=int(self.cp["DATA_PARAMETERS"]["img_count"])
        n=int(self.cp["DATA_PARAMETERS"]["n"])
        wr = [0.3,0.5]
        hr = [0.3,0.5]
        noise = 0.0
        flat=False
        cent=True
        if "wr" in self.cp["DATA_PARAMETERS"]:
            wr = self.__parse_range__(self.cp["DATA_PARAMETERS"]["wr"])
        if "hr" in self.cp["DATA_PARAMETERS"]:
            hr = self.__parse_range__(self.cp["DATA_PARAMETERS"]["hr"])
        if "noise" in self.cp["DATA_PARAMETERS"]:
            noise = float(self.cp["DATA_PARAMETERS"]["noise"])
        if "flat" in self.cp["DATA_PARAMETERS"]:
            flat = str_to_bool(self.cp["DATA_PARAMETERS"]["flat"])
        if "cent" in self.cp["DATA_PARAMETERS"]:
            cent = str_to_bool(self.cp["DATA_PARAMETERS"]["cent"])
        dg = DataGenerator()
        train, valid, test = dg.generate_images(img_count=img_count,n=n, wr=wr,hr=hr, noise=noise, cent=cent, flat=flat)
        return train, valid, test

    def __parse_range__(self, range_str):
        range = [float(x) for x in range_str[1:-1].split()]
        return range



def main():
    cp = NetworkParser("config.ini")
    network = cp.parse_network()
    #network = parse_config_network("config1.ini")
    print(cp.__parse_range__("(-0.1 0.1)"))
    
if __name__ == '__main__':
    main()