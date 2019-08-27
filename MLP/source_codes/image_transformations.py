import numpy as np
from skimage.feature import hog

def mean_normalize(x , _min, _max):
    mean = (_min+_max)/2
    return (x-mean)/mean

def add_noise_to_image(x, std_dev = 0.01):
    noise = np.random.normal(0, std_dev, np.shape(x))
    return x+noise

def hog_transform(image, number_of_bins = 10, pixels_per_cell_=(7,8)):
    return hog(image, orientations=number_of_bins, pixels_per_cell=pixels_per_cell_)

map_of_transforms = {"hog": hog_transform}

def transform_images(pixel_values, transform="hog", shape_of_reshaped_image=(-1)):
    rows = np.shape(pixel_values)[0]
    columns = np.shape(map_of_transforms[transform](np.reshape(pixel_values[0], (28,28))))[0]
    elements_to_return = np.zeros((rows, columns))
    for image_number in range(len(pixel_values)):
        image = np.reshape(pixel_values[image_number], (28,28))
        transformed_image = map_of_transforms[transform](image)
        elements_to_return[image_number] = np.reshape(transformed_image,shape_of_reshaped_image)
    return elements_to_return 

