import metalgpy as mpy
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds


(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# metalgpy
rng = np.random.RandomState(42)

Dense = mpy.meta(tf.keras.layers.Dense)
Model = mpy.meta(tf.keras.models.Model)


input_layer = tf.keras.layers.Input((28, 28))
x = tf.keras.layers.Flatten()(input_layer)

# layer with searched hyperparameters
x = Dense(
    mpy.Int(8, 16, name="units"),
    activation=mpy.List(["relu", "sigmoid"], name="activation"),
)(x)

output = Dense(10)(x)
model = Model(inputs=input_layer, outputs=output)

hyperparameters, model = list(mpy.sample(model, size=1, rng=rng))[0]
print(hyperparameters)

model = model.evaluate()

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
