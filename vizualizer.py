import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import sample_indices

class Visualizer:
    """
    class for vizualization of generated images
    """
    def show_n_random(self, images, n=10, flatened=True):
        """
        shows n random images from the set
        """
        indices = sample_indices(images, n)
        choices = images[indices]

        if flatened:
            dim = int(math.sqrt(len(images[0])))
            choices = [choices[i].reshape(dim,dim) for i in range(len(choices))]

        for img in choices:
            plt.matshow(img)
      
        plt.show()