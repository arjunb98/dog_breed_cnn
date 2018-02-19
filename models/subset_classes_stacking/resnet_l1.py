from keras.applications import ResNet50
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras import losses, optimizers
from keras.callbacks import LambdaCallback

class Resnet_l1():
    def __init__(self, name, classes, preTrained, trainingGenerator, validationGenerator):
        self.name = name
        self.classes = classes
        self.trainingGenerator = trainingGenerator
        self.validationGenerator = validationGenerator
        if preTrained:
            my_model = load_model("/models/"+name+".h5")
        else:
            my_model = Sequential()
            my_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
            #my_model.add(Dropout(.1))
            my_model.add(Dense(len(classes), activation='softmax', use_bias=True))

            my_model.layers[0].trainable = False
            sgd = optimizers.SGD(lr=0.01, momentum=.9)
            my_model.compile(optimizer=sgd, loss=losses.categorical_crossentropy, metrics=['accuracy'])
        self.model = my_model

    def fit(self, n_epochs=5, steps_per_epoch=68, val_steps=17, save=True):
        testmodelcb = LambdaCallback(on_epoch_end=self.testmodel)
        self.model.fit_generator(generator=self.trainingGenerator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=self.validationGenerator,
                            validation_steps=val_steps,
                            epochs=n_epochs,
                            callbacks=[testmodelcb],
                            verbose=1)
        if save:
            self.model.save(self.name+'.h5')

    def testmodel(self, epoch, logs):
        predx, predy = next(self.trainingGenerator)

        predout = self.model.predict(predx,batch_size=16)

        print("Input\n")
        print(predx)
        print("Target\n")
        print(predy)
        print("Prediction\n")
        print(predout)

    def predict(self,data):
        pass