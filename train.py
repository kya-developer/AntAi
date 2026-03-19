#train.py
import json
import tensorflow as tf
from model import create_model
import math

with open("trainingData.json") as f:
    data = json.load(f)
    
batch_size = 8

def gen():
    for item in data:
        features = {
            "tempurature": item["tempurature"],
            "humidity": item["humidity"],
            "light": item["light"],
            "time": item["time"],
            "species": item["species"],
        }
        label = item["behavior"]
        yield features, label

dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        {
            "tempurature": tf.TensorSpec((), tf.float32),
            "humidity": tf.TensorSpec((), tf.float32),
            "light": tf.TensorSpec((), tf.bool),
            "time": tf.TensorSpec((), tf.int32),
            "species": tf.TensorSpec((), tf.string),
        },
        tf.TensorSpec((), tf.string),
    )
).batch(batch_size)

species_lookUp = tf.keras.layers.StringLookup(output_mode="one_hot")
behavior_lookUp = tf.keras.layers.StringLookup(output_mode="one_hot")

# adapt lookups
species_lookUp.adapt(dataset.map(lambda x, y: x["species"]))
behavior_lookUp.adapt(dataset.map(lambda x, y: y))

dataset = dataset.map(lambda x, y: (x, behavior_lookUp(y)))
dataset = dataset.repeat()

with open("behavior_vocab.json", "w") as f:
    json.dump(behavior_lookUp.get_vocabulary(), f)


steps_per_epoch = math.ceil(len(data) / batch_size)
# create model after vocab is known
model, _, _ = create_model(species_lookUp, behavior_lookUp)

# compile & train
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(dataset, epochs=75, steps_per_epoch=steps_per_epoch)
model.save("ant_model.keras")