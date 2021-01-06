import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

modelo = load_model("comments.h5")

def predict(comment):
    return modelo.predict(np.array([comment]))[0][0]

while True:
    comment = input("comment: ")
    if comment == "q":
        break
    print(predict(comment) * 100, "por ciento")
