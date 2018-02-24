import os
import numpy as np
from PIL import Image
from random import shuffle, randint, seed
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.applications import ResNet50
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras import losses, optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from DataGenerator import DataGenerator
seed(42)
image_size = 224
num_classes = 120
validation_size = 0
pretrained_model = ''


def get_model(load_checkpoint):
    if load_checkpoint:
        print('loading model')
        return load_model(pretrained_model)
    my_model = Sequential()
    my_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    my_model.add(Dropout(.2))
    my_model.add(Dense(num_classes, activation='softmax', use_bias=True))

    my_model.layers[0].trainable = False
    sgd = optimizers.SGD(lr=0.001, momentum=.9)
    my_model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return my_model


img_paths_total = ['../input/train/'+x for x in os.listdir("../input/train")]
shuffle(img_paths_total)
labels_df = pd.read_csv('../input/labels.csv')
labels_list = list(set(labels_df['breed']))
labels_list.sort()
print(len(labels_list))
class_dict = {breed:labels_list.index(breed) for breed in labels_list}
labels_dict = {items[1]: items[2] for items in labels_df.itertuples()}
y = np.array([class_dict[labels_dict[path[15:-4]]] for path in img_paths_total])

validationFiles = img_paths_total[:validation_size]
validationLabels = dict((key, value) for (key, value) in zip(validationFiles, y[:validation_size]))
trainFiles = img_paths_total[validation_size:]
trainLabels = dict((key, value) for (key, value) in zip(trainFiles, y[validation_size:]))


partition = {'train': trainFiles, 'validation': validationFiles}
labels = {'train': trainLabels, 'validation': validationLabels}

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
          'random_location': False}
training_generator = DataGenerator(**paramsTrain).generate(labels['train'], partition['train'])
# validation_generator = DataGenerator(**paramsValid).generate(labels['validation'], partition['validation'])

callbacks_list = [ ModelCheckpoint('resnet_baseline.h5', monitor='val_acc', save_best_only=False, mode='max', period=2) ]

model = get_model(len(pretrained_model)>0)

model.fit_generator(generator = training_generator,
                        steps_per_epoch = len(partition['train'])//paramsTrain['batch_size'],
                        validation_steps = len(partition['validation'])//paramsValid['batch_size'],
                        epochs=10,
                        verbose=1)
model.save('resnet_baseline_epoch10_dr2_alldata.h5')
