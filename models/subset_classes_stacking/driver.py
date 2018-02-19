from random import shuffle
import numpy as np
import pandas as pd
import os

validation_size = .1
l1_models =[]
l2_models = []
num_l1_models = 12

def load_and_prep():
    img_paths_total = ['../../input/train/' + x for x in os.listdir("../../input/train")]
    labels_df = pd.read_csv('../../input/labels.csv')
    labels_list = list(set(labels_df['breed']))
    labels_list.sort()
    class_dict = {breed: labels_list.index(breed) for breed in labels_list}
    labels_dict = {items[1]: items[2] for items in labels_df.itertuples()}
    y = np.array([class_dict[labels_dict[path[15:-4]]] for path in img_paths_total])
    for i in range(num_l1_models):
        validationFiles = img_paths_total[:validation_size]
        validationLabels = dict((key, value) for (key, value) in zip(validationFiles, y[:validation_size]))
        trainFiles = img_paths_total[validation_size:]
        trainLabels = dict((key, value) for (key, value) in zip(trainFiles, y[validation_size:]))
        yield validationLabels, trainLabels