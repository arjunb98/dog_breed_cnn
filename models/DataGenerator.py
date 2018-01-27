# ## Custom Dataflow Generator
#
# This code is designed to take a random 512-512 patch for each epoch in order to minimize overfitting
#
# Code adapted from blog at: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

from random import randint
import numpy as np
from PIL import Image
from keras.applications.resnet50 import preprocess_input
class DataGenerator(object):

    def __init__(self, dim_x = 224, dim_y = 224, dim_z = 3, batch_size = 32, margin=100, shuffle = True, random_location = True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_location = random_location
        self.margin = margin

    def generate(self, labels, list_IDs):
        # Generates batches of samples
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                yield X, y

    def __get_exploration_order(self, list_IDs):
        # Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes


    def __data_generation(self, labels, list_IDs_temp):
        #Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size), dtype = int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            X[i, :, :, :] = read_and_crop(ID,margin=self.margin,random=self.random_location)

        # Store class
        y[i] = labels[ID]

        return preprocess_input(X), sparsify(y)

def read_and_crop(filepath, left=None, top=None, random = True, margin = 0, width = 224, height = 224):
    im_array = np.array(Image.open((filepath)), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    if left == None:
        if random:
            left = randint(margin, pil_im.size[0] - margin - width + 1)
        else:
            left = (pil_im.size[0] - width) // 2
    if top == None:
        if random:
            top = randint(margin, pil_im.size[1] - margin - height + 1)
        else:
            top = (pil_im.size[1] - height) // 2
    new_array = np.array(pil_im.crop((left,top,left+width,top+height)))
    return new_array

def sparsify(y):
    # Returns labels in binary NumPy array'
    n_classes = 102
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                   for i in range(y.shape[0])])