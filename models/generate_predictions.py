import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

ids = [x[:-4] for x in os.listdir("../input/test/test")]

labels_df = pd.read_csv('../input/labels.csv')
labels_list = list(set(labels_df['breed']))
labels_list.sort()
generator = ImageDataGenerator(preprocessing_function=preprocess_input)
pred_generator = generator.flow_from_directory('../input/test', class_mode=None, shuffle=False, batch_size=16, target_size=(224,224))
model = load_model("./resnet_baseline_epoch15.h5")
y = model.predict_generator(generator=pred_generator, steps=len(pred_generator), verbose=1)

submission = pd.DataFrame(data=y, columns=labels_list)
submission.to_csv('test_only_y.csv')
submission = pd.read_csv('test_only_y.csv',index_col=0)
submission.insert(0, 'id', ids)
submission.to_csv('test.csv', index=False)

