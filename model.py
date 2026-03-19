#model.py
import tensorflow as tf


def create_model(species_lookUp=None, behavior_lookUp=None):

    if species_lookUp is None:
        species_lookUp = tf.keras.layers.StringLookup(output_mode="one_hot", name="species_lookUp")
    if behavior_lookUp is None:
        behavior_lookUp = tf.keras.layers.StringLookup(output_mode="one_hot", name="behavior_lookUp")

    inputs = {
        "tempurature": tf.keras.Input(shape=(1,), name="tempurature"),
        "humidity": tf.keras.Input(shape=(1,), name="humidity"),
        "light": tf.keras.Input(shape=(1,), name="light"),
        "time": tf.keras.Input(shape=(1,), name="time"),
        "species": tf.keras.Input(shape=(), dtype=tf.string, name="species"),
    }

    species_encoded = species_lookUp(inputs["species"])
    light_numeric = tf.keras.layers.Activation("linear", dtype=tf.float32)(
        inputs["light"]
    )

    features = tf.keras.layers.Concatenate()([
        inputs["tempurature"],
        inputs["humidity"],
        light_numeric,
        inputs["time"],
        species_encoded
    ])

    x = tf.keras.layers.Dense(64, activation="relu")(features)
    x = tf.keras.layers.Dense(32, activation="relu")(x)

    output = tf.keras.layers.Dense(
        len(behavior_lookUp.get_vocabulary()),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model, species_lookUp, behavior_lookUp
