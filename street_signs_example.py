from my_utils import split_data, order_test_set
from PIL.Image import FASTOCTREE
from deepLearningModels import streetsigns_model
from my_utils import create_generators
import tensorflow



if __name__ == '__main__':

    if False:
      path_to_data = '/Users/bryanirwin/Downloads/archive/Train'
      path_to_save_train = '/Users/bryanirwin/Downloads/archive/train_data/train'
      path_to_save_val = '/Users/bryanirwin/Downloads/archive/train_data/val'
      split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)
    
    if False:
      path_to_images = '/Users/bryanirwin/Downloads/archive/Test'
      path_to_csv = '/Users/bryanirwin/Downloads/archive/Test.csv'
      order_test_set(path_to_images, path_to_csv)

    path_to_train = '/Users/bryanirwin/Downloads/archive/train_data/train'
    path_to_val = '/Users/bryanirwin/Downloads/archive/train_data/val'
    path_to_test = '/Users/bryanirwin/Downloads/archive/Test'
    batch_size=64
    epochs=15

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    

TRAIN=False
TEST=True

if TRAIN:
    path_to_save_model = './Models'
    ckpt_saver = tensorflow.keras.callbacks.ModelCheckpoint(
        path_to_save_model,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )

    early_stop = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10
    )

    model = streetsigns_model(train_generator.num_classes)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=val_generator,
        callbacks=[ckpt_saver, early_stop] 
        )

if TEST:
    model = tensorflow.keras.models.load_model('./Models')
    model.summary()
    print('evaluating validation set')
    model.evaluate(val_generator)
    print('evaluating test set')
    model.evaluate(test_generator)
