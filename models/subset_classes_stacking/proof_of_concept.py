import numpy as np
from random import shuffle, randint, seed
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from DataGenerator import DataGenerator
from resnet_l1 import Resnet_l1

seed(42)
SEED = 42
image_size = 224
num_classes = 12
pretrained_model = ''

labels = pd.read_csv('../../input/labels.csv')
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
labels['rank'] = labels.groupby('breed').rank()['id']
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
labels['id'] = labels['id'].apply(lambda x: "../../input/train/"+x+".jpg")

np.random.seed(seed=SEED)
rnd = np.random.random(len(labels))
train_idx = rnd < 0.8
valid_idx = rnd >= 0.8
y_train = labels_pivot[selected_breed_list].values
x_train = labels['id'].values

ytr = y_train[train_idx]
yv = y_train[valid_idx]
xtr= x_train[train_idx]
xv = x_train[valid_idx]

partition = {'train': xtr, 'validation': xv}
labels = {'train': ytr, 'validation': yv}

paramsTrain = {'dim_x': 224,
          'dim_y': 224,
          'dim_z': 3,
          'batch_size': 16,
          'shuffle': True,
          'margin': 100,
          'random_location': True}
paramsValid = {'dim_x': 224,
          'dim_y': 224,
          'dim_z': 3,
          'batch_size': 16,
          'shuffle': True,
          'margin': 0,
          'random_location': True}
training_generator = DataGenerator(**paramsTrain).generate(labels['train'], partition['train'])
validation_generator = DataGenerator(**paramsValid).generate(labels['validation'], partition['validation'])

model = Resnet_l1('proof_of_concept',selected_breed_list, False, training_generator,validation_generator)
model.fit()
