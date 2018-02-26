import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from random import shuffle, randint, seed
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.applications.xception import preprocess_input as preprocess_input_xception
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from keras.models import load_model
from keras.applications import ResNet50
from keras.applications import Xception
from keras.applications import InceptionV3
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score


def create_l1_models():
    models = []
    resnet = Sequential()
    resnet.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    models.append(resnet)
    xception = Sequential()
    xception.add(Xception(include_top=False, pooling='avg', weights='imagenet'))
    models.append(xception)
    preprocessors = [preprocess_input_resnet, preprocess_input_xception, preprocess_input_inception]
    inception = Sequential()
    inception.add(InceptionV3(include_top=False, pooling='avg', weights='imagenet'))
    models.append(inception)
    return models, preprocessors


def generate_features(models, preprocessers):
    all_features=[]
    for i in range(len(models)):
        generator = ImageDataGenerator(preprocessing_function=preprocessers[i])
        pred_generator = generator.flow_from_directory('../../input/train', class_mode=None, shuffle=False, batch_size=64,
                                                       target_size=(224, 224))
        all_features.append(models[i].predict_generator(generator=pred_generator, steps=len(pred_generator), verbose=1))
    return np.concatenate(all_features,axis=-1)


def train_l2(features, labels, val=False):
    # for c in [.01,.02,.03,.04,.05,.06,.07,.08,.1]:
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=.03)
    model.fit(features['train'],labels['train'])
    if val:
        probs = model.predict_proba(features['validation'])
        preds = model.predict(features['validation'])
        # print('C = ', c)
        print('Validation LogLoss {}'.format(log_loss(labels['validation'], probs)))
        print('Validation Accuracy {}'.format(accuracy_score(labels['validation'], preds)))
    return model

def generate_predictions(models, preprocessers, level2):
    all_features = []
    for i in range(len(models)):
        generator = ImageDataGenerator(preprocessing_function=preprocessers[i])
        pred_generator = generator.flow_from_directory('../../input/test', class_mode=None, shuffle=False,
                                                       batch_size=64,
                                                       target_size=(224, 224))
        all_features.append(models[i].predict_generator(generator=pred_generator, steps=len(pred_generator), verbose=1))
    X = np.concatenate(all_features, axis=-1)
    save2 = pd.DataFrame(X)
    save2.to_csv('savePred.csv',index_label=False)
    y = level2.predict_proba(X)
    ids = [x[:-4] for x in os.listdir("../../input/test/test")]
    labels_df = pd.read_csv('../../input/labels.csv')
    labels_list = list(set(labels_df['breed']))
    labels_list.sort()
    submission = pd.DataFrame(data=y, columns=labels_list)
    submission.to_csv('temp.csv')
    submission = pd.read_csv('temp.csv', index_col=0)
    submission.insert(0, 'id', ids)
    submission.to_csv('logreg_imagenet_features_v2.csv', index=False)


if __name__ == '__main__':
    models, preprocess = create_l1_models()
    # X = generate_features(models,preprocess)
    # labels_df = pd.read_csv('../../input/labels.csv')
    # labels_list = list(set(labels_df['breed']))
    # save = pd.DataFrame(X)
    # save.to_csv('save.csv', index_label=False)
    X = pd.read_csv('save.csv')
    X.drop('0',axis=1)
    labels = pd.read_csv('../../input/labels.csv')
    labels['target'] = 1
    labels['rank'] = labels.groupby('breed').rank()['id']
    labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
    labels['id'] = labels['id'].apply(lambda x: "../../input/train/" + x + ".jpg")

    np.random.seed(seed=420)
    rnd = np.random.random(len(labels))
    train_idx = rnd < 2
    valid_idx = rnd >= 2
    y_train = np.array([y[1:] for y in labels_pivot.values])
    y_train = np.array([y_train[i].nonzero()[0] for i in range(len(y_train))])
    y_train = y_train.flatten()
    print(y_train)
    x_train = X

    ytr = y_train[train_idx]
    yv = y_train[valid_idx]
    xtr = x_train[train_idx]
    xv = x_train[valid_idx]
    partition = {'train': xtr, 'validation': xv}
    labels = {'train': ytr, 'validation': yv}

    model = train_l2(partition,labels,val=False)
    generate_predictions(models, preprocess, model)