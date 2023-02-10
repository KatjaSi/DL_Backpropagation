import math
from doodler_forall import gen_standard_cases

class DataGenerator():
    """
    Class for generating images
    """

    def generate_images(self, n, img_count=20,  noise=0, wr=[0.2,0.4],hr=[0.2,0.5], cent=False, flat=False,
                        train_frac=0.7, valid_frac=0.2, test_frac=0.1):
        """ Method for generating unique images

        Parameters
        ----------
        n : int
            The dimention of each image nxn
        img_count : int
            The number of images to be generated
        noise : float
            Probability that any given pixel will be flipped from foreground to background or vice versa.
        wr : float
            The width range, which is a fraction of the total width of the canvas 
        hr : float
            The heigh range, which is a fraction of the total high of the canvas   
        cent : bool
            Defines whether the object should be centered
        flat : bool 
            Defines whether generated images to be returned as flat vectors or 2-d arrays
        train_frac, valid_frac, test_frac : float
            The fractions of the images in training, validation and test sets

        Returns
        -------
        three sets of images: train, validation and test. Each set is a 5-item tuple: (images, targets, labels, 2d-image-dimensions, flat)

        """

        assert n >= 10 and n <= 50, "n should be a value between 10 and 50'"
        assert math.isclose(train_frac+valid_frac+test_frac, 1.0), "fractions for train, validation and test sets should sum to 1.0'"
        cases = gen_standard_cases(count=img_count, rows=n, cols=n, types=['ball','box','triangle','bar'],
         show=False, wr=wr, hr=hr, noise=noise, cent=cent, flat=flat)
        # dividing cases into train, valid and test sets
        train_set_count = int(img_count*train_frac)
        valid_set_count = int(img_count*valid_frac)
        train_set = (cases[0][:train_set_count], 
                     cases[1][:train_set_count], 
                     cases[2][:train_set_count], 
                     cases[3],
                     cases[4])
        valid_set = (cases[0][train_set_count:train_set_count+valid_set_count], 
                     cases[1][train_set_count:train_set_count+valid_set_count], 
                     cases[2][train_set_count:train_set_count+valid_set_count], 
                     cases[3],
                     cases[4])
        test_set = (cases[0][train_set_count+valid_set_count:], 
                     cases[1][train_set_count+valid_set_count:], 
                     cases[2][train_set_count+valid_set_count:], 
                     cases[3],
                     cases[4])
        return train_set, valid_set, test_set 
