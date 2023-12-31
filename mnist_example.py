import tensorflow
from deepLearningModels import functional_model, MyCustomModel
from my_utils import display_some_examples

#tensorflow.keras.Sequential
seq_model = tensorflow.keras.Sequential(
    [
        tensorflow.keras.layers.Input(shape = (28,28,1)),
        tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tensorflow.keras.layers.MaxPool2D(),
        tensorflow.keras.layers.BatchNormalization(),

        tensorflow.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tensorflow.keras.layers.BatchNormalization(),
        tensorflow.keras.layers.MaxPool2D(),
        
        tensorflow.keras.layers.GlobalAveragePooling2D(),
        tensorflow.keras.layers.Dense(64, activation='relu'),
        tensorflow.keras.layers.Dense(10, activation='softmax')
    ]
)

if __name__=='__main__':
    
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    
    print('x_train.shape =', x_train.shape)
    print('y_train.shape =', y_train.shape)
    print('x_test.shape =', x_test.shape)
    print('y_test.shape =', y_test.shape)

    if False:
      display_some_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    x_train = np.expand_dims(x_train, axis=-1)
    x_test =  np.expand_dims(x_test, axis=-1)

    y_train = tensorflow.keras.utils.to_categorical(y_train)
    y_test = tensorflow.keras.utils.to_categorical(y_test)

    # model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    
    #model training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    
    #Evaluation on test dataset
    model.evaluate(x_test, y_test, batch_size=64)