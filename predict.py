# predict.py
import tensorflow as tf
import numpy as np
import json

# Load trained model
model = tf.keras.models.load_model("ant_model.keras")

# Load the behavior vocabulary
with open("behavior_vocab.json") as f:
    behavior_vocab = json.load(f)


def Predict(tempurature, humidity, light, time, species):
    
    input_data = {
        "tempurature": tf.constant([tempurature], dtype=tf.float32),
        "humidity": tf.constant([humidity], dtype=tf.float32),
        "light": tf.constant([light], dtype=tf.bool),
        "time": tf.constant([time], dtype=tf.int32),
        "species": tf.constant([species], dtype=tf.string),
    }

    preditions = model.predict(input_data)
    pred_index = int(np.argmax(preditions[0]))
    pred_label = behavior_vocab[pred_index]

    return pred_label