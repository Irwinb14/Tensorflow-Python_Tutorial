import tensorflow
import matplotlib.pyplot as plt
import numpy as np

# functional approach : function that returns a deep learning model
def functional_model():
        my_input = tensorflow.keras.layers.Input(shape = (28,28,1))
        x = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu')(my_input)
        x = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tensorflow.keras.layers.MaxPool2D()(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)

        x = tensorflow.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        x = tensorflow.keras.layers.MaxPool2D()(x)
        
        
        x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
        x = tensorflow.keras.layers.Dense(64, activation='relu')(x)
        x = tensorflow.keras.layers.Dense(10, activation='softmax')(x)

        model = tensorflow.keras.Model(inputs=my_input, outputs=x)

        return model

# tensorflow.keras.Model : inherit from this class 
class MyCustomModel(tensorflow.keras.Model):

    def __init__(self):
        super().__init__()

        self.conv1 = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu')        
        self.maxpool1 = tensorflow.keras.layers.MaxPool2D()
        self.batchnorm1 = tensorflow.keras.layers.BatchNormalization()

        self.conv3 = tensorflow.keras.layers.Conv2D(128, (3,3), activation='relu')
        self.batchnorm2 = tensorflow.keras.layers.BatchNormalization()
        self.maxpool2 = tensorflow.keras.layers.MaxPool2D()
        
        
        self.globalaveragepool1 = tensorflow.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tensorflow.keras.layers.Dense(64, activation='relu')
        self.dense2 = tensorflow.keras.layers.Dense(10, activation='softmax')

    def call(self, my_input):

        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.globalaveragepool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


def streetsigns_model(nbr_classes):

    my_input = tensorflow.keras.layers.Input(shape=(60, 60, 3))

    x = tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu')(my_input)
    x = tensorflow.keras.layers.MaxPool2D()(x)
    x = tensorflow.keras.layers.BatchNormalization()(x) 
    
    x = tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tensorflow.keras.layers.MaxPool2D()(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    x = tensorflow.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tensorflow.keras.layers.MaxPool2D()(x)
    x = tensorflow.keras.layers.BatchNormalization()(x)

    # x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(128, activation='relu')(x)
    x = tensorflow.keras.layers.Dense(nbr_classes, activation='softmax')(x)

    model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return model


if __name__=='__main__':
    model = streetsigns_model(10)
    model.summary()