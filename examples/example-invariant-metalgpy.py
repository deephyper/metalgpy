#! MetalgPy
import metalgpy as mpy
import numpy as np

rng = np.random.RandomState(42)

max_layers = 5
num_layers = mpy.Int(1, max_layers, name="num_layers")
layers = (
    ["input_layer"]
    + mpy.List(
        values=[
            mpy.List([f"dense_{i}", f"conv_{i}"], name=f"layer_{i}")
            for i in range(max_layers)
        ],
        k=num_layers,
        invariant=True,
        name="layers",
    )
    + ["output_layer"]
)
print(layers)

variables = layers.variables()
print(variables)


sample_choice, sample_exp = list(mpy.sample(layers, size=1, rng=rng))[0]
print(sample_choice)
print(sample_exp)
res = sample_exp.evaluate()
print(res)
