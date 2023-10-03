import tensorflow as tf
import numpy as np

def predict_with_model(model, img_path):
  
  image = tf.io.read_file(img_path)
  image = tf.image.decode_png(image, channels=3)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.resize(image, [60, 60])
  image = tf.expand_dims(image, axis=0)

  predictions = model.predict(image) #gives a weighted value for each image class
  predictions = np.argmax(predictions) #gives the index(class) for the highest prediction value

  return predictions

if __name__=='__main__':
    # img_path = '/Users/bryanirwin/Downloads/archive/Test/2/00092.png'
    img_path = '/Users/bryanirwin/Downloads/archive/Test/0/00579.png'
    model= tf.keras.models.load_model('./Models')
    prediction =  predict_with_model(model, img_path) 

    print(f'prediction = {prediction}')