from google.colab import drive
import tensorflow as tf


def get_model():
    drive.mount('/content/gdrive')
    model = tf.keras.models.load_model('my_model.h5')
    return model
# Show the model architecture
get_model().summary()

