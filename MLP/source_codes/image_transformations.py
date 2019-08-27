import numpy as np
from skimage.feature import hog

map_of_transforms = {"hog":hog_transform}

def mean_normalize(x , _min, _max):
    mean = (_min+_max)/2
    return (x-mean)/mean

def add_noise_to_image(x, std_dev = 0.01):
    noise = np.random.normal(0, std_dev, np.shape(x))
    return x+noise

def hog_transform(image, number_of_bins = 10, pixels_per_cell_=(7,8)):
    return hog(image, orientations=number_of_bins, pixels_per_cell=pixels_per_cell_)

def transform_images(pixel_values, transform="hog", shape_of_reshaped_image=(-1)):
    for image_number in range(len(pixel_values)):
        image = np.reshape(image, (28,28))
        transformed_image = map_of_transforms[transform](image)
        pixel_values[image_number] = np.reshape(transformed_image,shape_of_reshaped_image)
    return pixel_values    