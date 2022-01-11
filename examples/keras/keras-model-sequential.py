import tensorflow as tf
import metalgpy as mpy
import numpy as np

# meta symbols
Sequential = mpy.meta(tf.keras.models.Sequential)
Dense = mpy.meta(tf.keras.layers.Dense)
Lambda = mpy.meta(tf.keras.layers.Lambda)

# meta program
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    *[mpy.List([
        Lambda(lambda x: x), 
        Dense(mpy.Int(8, 32), activation="relu")
        ]) for _ in range(5)
    ],
    tf.keras.layers.Dense(10)
])

# search space
choices = model.choice()

rng = np.random.RandomState(42)

for choice, model in mpy.sample(model, size=5, rng=rng, deepcopy=True):
    print(" ** Sampling new model: ")

    print("    - choice: ", choice, end="\n\n")

    model = model.evaluate()
    model.summary()


    print("\n"*3)